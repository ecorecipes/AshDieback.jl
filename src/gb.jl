# Great Britain: 10 km squares of the OS National Grid, with ash area, first finds and temperature
# (data/gb/README.md). Nothing here is fitted to GB.
using Random: Xoshiro

"""
    GBInputs

The 10 km squares of Great Britain with land, from [`load_gb_inputs`](@ref).

- `squares`: `(easting_km, northing_km)` of each square's south-west corner.
- `ash_ha`: modelled ash area in the square, in hectares.
- `seeded`: whether ash dieback was first confirmed in the square in 2012–2015. The model's
  epidemic starts in these squares.
- `t28_years`, `t28`: days over 28 °C in July and August from HadUK-Grid, squares × years.
"""
struct GBInputs
    squares::Vector{Tuple{Int,Int}}
    ash_ha::Vector{Float64}
    seeded::Vector{Bool}
    t28_years::UnitRange{Int}
    t28::Matrix{Float64}
end

function read_csv_rows(path::AbstractString)
    isfile(path) || throw(DataFormatError(path, "file not found"))
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && throw(DataFormatError(path, "no header row"))
    return split(lines[1], ','), [split(l, ',') for l in lines[2:end]]
end

"""
    load_gb_inputs(dir = joinpath(datadir(), "gb")) -> GBInputs

Read Great Britain's squares, ash area, seeded squares and observed temperature. See
`data/gb/README.md` for sources and licences.
"""
function load_gb_inputs(dir::AbstractString = joinpath(datadir(), "gb"))
    _, ash = read_csv_rows(joinpath(dir, "ash_10km.csv"))
    squares = [(parse(Int, r[1]), parse(Int, r[2])) for r in ash]
    ash_ha = [parse(Float64, r[3]) for r in ash]
    _, sd = read_csv_rows(joinpath(dir, "seed_squares_10km.csv"))
    seeds = Set((parse(Int, r[1]), parse(Int, r[2])) for r in sd)
    issubset(seeds, squares) ||
        throw(DataFormatError(joinpath(dir, "seed_squares_10km.csv"), "a seeded square is not a land square"))
    head, tt = read_csv_rows(joinpath(dir, "t28_haduk_10km.csv"))
    years = parse(Int, head[3][2:end]):parse(Int, head[end][2:end])
    length(years) == length(head) - 2 ||
        throw(DataFormatError(joinpath(dir, "t28_haduk_10km.csv"), "years are not consecutive"))
    length(tt) == length(squares) ||
        throw(DataFormatError(joinpath(dir, "t28_haduk_10km.csv"), "not one row per square"))
    t28 = zeros(length(squares), length(years))
    for (i, r) in enumerate(tt)
        (parse(Int, r[1]), parse(Int, r[2])) == squares[i] ||
            throw(DataFormatError(joinpath(dir, "t28_haduk_10km.csv"), "row $(i + 1) is not in the order of ash_10km.csv"))
        t28[i, :] .= parse.(Float64, view(r, 3:length(r)))
    end
    return GBInputs(squares, ash_ha, [s in seeds for s in squares], years, t28)
end

"""
    load_gb_future(inputs::GBInputs; dir = joinpath(datadir(), "gb")) -> Dict

Days over 28 °C from CHESS-SCAPE, bias-corrected, for model years 2026–2080, rounded to 0.01 days:
`(scenario, member) => squares × years`, for `"rcp45"` and `"rcp85"` and members `"01"`, `"04"`,
`"06"`, `"15"`. The few squares without a value take their nearest square's. CHESS-SCAPE years are
model years, not real ones, and RCP4.5 is scaled from RCP8.5.
"""
function load_gb_future(inputs::GBInputs; dir::AbstractString = joinpath(datadir(), "gb"))
    nsq, ny = length(inputs.squares), 55
    path = joinpath(dir, "t28_chess_2026_2080.int16")
    isfile(path) || throw(DataFormatError(path, "file not found"))
    raw = Vector{Int16}(undef, 8 * nsq * ny)
    read!(path, raw)
    raw .= ltoh.(raw)
    has = [raw[(i - 1) * ny + 1] >= 0 for i in 1:nsq]
    known = findall(has)
    nearest = [has[i] ? i :
               known[argmin([(inputs.squares[k][1] - inputs.squares[i][1])^2 +
                             (inputs.squares[k][2] - inputs.squares[i][2])^2 for k in known])]
               for i in 1:nsq]
    out = Dict{Tuple{String,String},Matrix{Float64}}()
    n = 0
    for scenario in ("rcp45", "rcp85"), member in ("01", "04", "06", "15")
        block = view(raw, (n * nsq * ny + 1):((n + 1) * nsq * ny))
        m = permutedims(reshape(block, ny, nsq)) ./ 100
        out[(scenario, member)] = m[nearest, :]
        n += 1
    end
    return out
end

"""
    gb_problem(inputs::GBInputs; B = 25.0, theta, temperature = nothing, last_year = 2100,
               final_year = last_year, initialization_year = 2014, border = 10, seed = 1,
               delta_t = 1.0) -> (prob, cell)

The model on Great Britain, with no survey: an [`AshDiebackProblem`](@ref) to pass to
[`HostProblem`](@ref), and `cell(easting_km, northing_km)`, which gives a square's `(i, j)`.

- Ash density in a square is `ash_ha / 100 / 100 * B`, with `B` the stand basal area in m²/ha.
- The epidemic is seeded in `initialization_year` in the squares first found in 2012–2015, with
  `theta`'s `C_init` as their share of symptomatic trees.
- `theta` defaults to the paper's θmax with `C_init = 0.05`.
- Temperature is HadUK-Grid to 2025. After that it is whole years resampled from 2004–2025, or
  with `temperature` (a matrix from [`load_gb_future`](@ref)) that matrix to 2080, then whole years
  resampled from its 2071–2080. Years past 2080 are for the code's horizon, not results.
"""
function gb_problem(inputs::GBInputs; B::Real = 25.0,
                    theta = [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.05, 0.056],
                    temperature = nothing, last_year::Integer = 2100, final_year::Integer = last_year,
                    initialization_year::Integer = 2014, border::Integer = 10, seed::Integer = 1,
                    delta_t::Real = 1.0)
    es, ns = first.(inputs.squares), last.(inputs.squares)
    e0, n0 = minimum(es), minimum(ns)
    grid = QuadratGrid(; quadrat_length = 10.0, nx_mainland = (maximum(es) - e0) ÷ 10 + 1,
                       ny_mainland = (maximum(ns) - n0) ÷ 10 + 1, border = (border, border, border, border))
    nx, ny = size(grid)
    cell(e, n) = ((e - e0) ÷ 10 + border + 1, (n - n0) ÷ 10 + border + 1)
    y0 = first(inputs.t28_years)
    years = y0:last_year
    T = zeros(nx, ny, length(years))
    d, ψ = zeros(nx, ny), zeros(nx, ny)
    for (k, s) in enumerate(inputs.squares)
        i, j = cell(s...)
        T[i, j, 1:length(inputs.t28_years)] .= view(inputs.t28, k, :)
        d[i, j] = inputs.ash_ha[k] / 100 / 100 * B
        inputs.seeded[k] && (ψ[i, j] = 1.0)
    end
    rng = Xoshiro(seed)
    last_observed = last(inputs.t28_years)
    if temperature === nothing
        for y in (last_observed + 1):last_year
            T[:, :, y - y0 + 1] .= view(T, :, :, rand(rng, 1:length(inputs.t28_years)))
        end
    else
        for (k, s) in enumerate(inputs.squares), y in 2026:min(2080, last_year)
            i, j = cell(s...)
            T[i, j, y - y0 + 1] = temperature[k, y - 2025]
        end
        for y in 2081:last_year
            T[:, :, y - y0 + 1] .= view(T, :, :, rand(rng, 2071:2080) - y0 + 1)
        end
    end
    obs = [YearObservations(a, Int[], Int[], Float64[], Int[], Int[], Int[], Float64[]) for a in (y0 + 1):last_year]
    data = PEOCData(grid, years, zeros(nx, ny, length(years)), T, :T28, d, (ψ, 2 .* ψ), obs)
    settings = ModelSettings(; initialization_year, final_year, delta_t)
    return AshDiebackProblem(data, theta; settings), cell
end
