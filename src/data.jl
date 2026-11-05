"""
    YearObservations

The disease survey of one year, aggregated by quadrat.

Fields:
- `year::Int`: the survey year.
- `x::Vector{Int}`, `y::Vector{Int}`: grid indices of the visited quadrats.
- `proportion::Vector{Float64}`: mean proportion of trees with symptoms over the visits.
- `visits::Vector{Int}`: number of plot visits; each visit observes `observed_trees` trees.
- `visit_x`, `visit_y`, `visit_proportion`: the individual plot visits behind the means, in
  file order, as [`rmse`](@ref) uses them.

Repeat visits to a quadrat are merged in file order with a running mean, as PEOC does.
"""
struct YearObservations
    year::Int
    x::Vector{Int}
    y::Vector{Int}
    proportion::Vector{Float64}
    visits::Vector{Int}
    visit_x::Vector{Int}
    visit_y::Vector{Int}
    visit_proportion::Vector{Float64}
end

Base.length(o::YearObservations) = length(o.x)

"""
    PEOCData

The input data of the model on a [`QuadratGrid`](@ref), as read by [`load_peoc_data`](@ref).

Fields:
- `grid::QuadratGrid`
- `climate_years::UnitRange{Int}`: the years of `rainfall` and `temperature` (third index).
- `rainfall::Array{Float64,3}`: June rainfall h_a (mm), size `(nx, ny, length(climate_years))`.
- `temperature::Array{Float64,3}`: the temperature index T_a (days in July and August with a
  maximum over the threshold), same size.
- `temperature_index::Symbol`: which index `temperature` holds (`:T24`, `:T26`, `:T28`,
  `:T30` or `:T35`).
- `ash_density::Matrix{Float64}`: smoothed ash basal area d (m²/ha); zero in the border.
- `psi::NTuple{2,Matrix{Float64}}`: the smoothed prevalence maps Ψ used to impute the
  colonised rachis of the two years before the first simulated year.
- `observations::Vector{YearObservations}`: the survey, one entry per year.
"""
struct PEOCData
    grid::QuadratGrid
    climate_years::UnitRange{Int}
    rainfall::Array{Float64,3}
    temperature::Array{Float64,3}
    temperature_index::Symbol
    ash_density::Matrix{Float64}
    psi::NTuple{2,Matrix{Float64}}
    observations::Vector{YearObservations}
end

"""
    datadir()

Directory of the data files shipped with the package: the PEOC 1.0 release data
(`climate_data`, `density.txt`, `data_chalara.txt`, `rachis_initialization.txt`).
"""
datadir() = normpath(joinpath(@__DIR__, "..", "data"))

const TEMPERATURE_COLUMNS = (T24 = 12, T26 = 13, T28 = 10, T30 = 14, T35 = 15)

# Rows of a whitespace-separated file with a header, as numpy.loadtxt(dtype=str) reads it:
# any line ending (density.txt uses CR only), blank lines skipped, header dropped.
function read_table(path::AbstractString)
    isfile(path) || throw(DataFormatError(path, "file not found"))
    lines = split(read(path, String), r"\r\n|\r|\n")
    rows = [split(l) for l in lines if !isempty(strip(l))]
    isempty(rows) && throw(DataFormatError(path, "no header row"))
    return rows[2:end]
end

function column(rows, path, j::Integer)
    return map(enumerate(rows)) do (r, row)
        j <= length(row) ||
            throw(DataFormatError(path, "row $(r + 1) has no column $j"))
        row[j]
    end
end

function float_column(rows, path, j::Integer)
    return map(enumerate(column(rows, path, j))) do (r, s)
        v = tryparse(Float64, s)
        v === nothing &&
            throw(DataFormatError(path, "row $(r + 1), column $j: $(repr(s)) is not a number"))
        v
    end
end

function checked_index(grid, path, x, y)
    ij = quadrat_index(grid, x, y)
    ij === nothing && return nothing
    (1 <= ij[1] <= size(grid, 1) && 1 <= ij[2] <= size(grid, 2)) ||
        throw(DataFormatError(path, "point ($x, $y) is outside the grid"))
    return ij
end

function read_climate(path, grid, years, column_index)
    rows = read_table(path)
    x = float_column(rows, path, 1) .* 1000  # km -> m
    y = float_column(rows, path, 2) .* 1000
    year = float_column(rows, path, 3)
    value = float_column(rows, path, column_index)
    # PEOC fills with -1, assigns the data and then sets every -1 to 0.
    X = fill(-1.0, size(grid)..., length(years))
    for r in eachindex(x)
        ij = checked_index(grid, path, x[r], y[r])
        (ij === nothing || year[r] < first(years)) && continue
        a = Int(year[r]) - first(years) + 1
        a <= length(years) ||
            throw(DataFormatError(path, "year $(Int(year[r])) is after $(last(years))"))
        X[ij..., a] = value[r]
    end
    X[X .== -1] .= 0
    return X
end

function read_density(path, grid)
    rows = read_table(path)
    x, y = float_column(rows, path, 1), float_column(rows, path, 2)
    d = float_column(rows, path, length(first(rows)))
    density = zeros(size(grid))
    for r in eachindex(x)
        ij = checked_index(grid, path, x[r], y[r])
        ij === nothing || (density[ij...] = d[r])
    end
    return density
end

function read_initialization(path, grid)
    rows = read_table(path)
    # PEOC keeps a row only if the *string* in the last used column is not "0".
    rows = rows[column(rows, path, 10) .!= "0"]
    x = float_column(rows, path, 2) ./ 1000  # stored 1000 times too large
    y = float_column(rows, path, 3) .* 1000  # km -> m
    psi1, psi2 = float_column(rows, path, 9), float_column(rows, path, 10)
    P1, P2 = zeros(size(grid)), zeros(size(grid))
    for r in eachindex(x)
        ij = checked_index(grid, path, x[r], y[r])
        ij === nothing && continue
        P1[ij...] = psi1[r]
        P2[ij...] = psi2[r]
    end
    return P1, P2
end

function read_observations(path, grid, years)
    rows = read_table(path)
    year = float_column(rows, path, 4)
    xi = float_column(rows, path, 7) .+ grid.border[1]
    yi = float_column(rows, path, 8) .+ grid.border[3]
    preval = float_column(rows, path, 11)
    return map(years) do a
        keys = Tuple{Int,Int}[]
        p = Float64[]
        c = Int[]
        position = Dict{Tuple{Int,Int},Int}()
        rows_a = findall(==(a), year)
        for r in rows_a
            key = (Int(xi[r]) + 1, Int(yi[r]) + 1)
            k = get(position, key, 0)
            if k == 0
                push!(keys, key)
                push!(p, preval[r])
                push!(c, 1)
                position[key] = length(keys)
            else  # running mean, evaluated as in PEOC
                p[k] = (p[k] * c[k] + preval[r] * 1) / (c[k] + 1)
                c[k] += 1
            end
        end
        YearObservations(a, first.(keys), last.(keys), p .* 0.01, c,
                         Int.(xi[rows_a]) .+ 1, Int.(yi[rows_a]) .+ 1, preval[rows_a] .* 0.01)
    end
end

"""
    load_peoc_data(dir = datadir(); temperature_index = :T28, climate_years = 2007:2022,
                   observation_years = 2008:2023, grid = QuadratGrid())

Read the PEOC input files from `dir` and put them on `grid`.

Reproduces PEOC's parsing exactly: coordinates are rounded to quadrats with ties to even,
Corsica is dropped, cells without data are zero, survey quadrat indices are shifted by the
west and south borders, and repeat visits to a quadrat are merged in file order. The arrays
are bitwise equal to PEOC's.

`temperature_index` chooses the climate column used for T: `:T24`, `:T26`, `:T28` (the
paper's choice), `:T30` or `:T35`.
"""
function load_peoc_data(dir::AbstractString = datadir(); temperature_index::Symbol = :T28,
                        climate_years::UnitRange{Int} = 2007:2022,
                        observation_years::UnitRange{Int} = 2008:2023,
                        grid::QuadratGrid = QuadratGrid())
    haskey(TEMPERATURE_COLUMNS, temperature_index) ||
        throw(ArgumentError("temperature_index must be one of " *
                            "$(keys(TEMPERATURE_COLUMNS)), not $(repr(temperature_index))"))
    climate = joinpath(dir, "climate_data")
    H = read_climate(climate, grid, climate_years, 6)
    T = read_climate(climate, grid, climate_years, TEMPERATURE_COLUMNS[temperature_index])
    d = read_density(joinpath(dir, "density.txt"), grid)
    psi = read_initialization(joinpath(dir, "rachis_initialization.txt"), grid)
    obs = read_observations(joinpath(dir, "data_chalara.txt"), grid, observation_years)
    return PEOCData(grid, climate_years, H, T, temperature_index, d, psi, obs)
end

function Base.show(io::IO, ::MIME"text/plain", d::PEOCData)
    nx, ny = size(d.grid)
    println(io, "PEOCData on a $nx × $ny grid of $(d.grid.quadrat_length) km quadrats")
    println(io, "  climate years:     ", d.climate_years, " (temperature index ",
            d.temperature_index, ")")
    println(io, "  ash density > 0:   ", count(>(0), d.ash_density), " quadrats")
    n = sum(length, d.observations)
    print(io, "  survey:            ", first(d.observations).year, ":",
          last(d.observations).year, ", ", n, " quadrat-years")
end
