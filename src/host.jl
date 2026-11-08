"""
    HostClass(name; share = 1.0, susceptibility = 1.0, mortality = 0.0, inoculum = 1.0,
              offspring = name)

One kind of ash in a [`HostDynamics`](@ref).

- `share`: its share of the host at the start, and of recruits not inheriting from a parent.
- `susceptibility`: s, multiplying the yearly chance of developing symptoms.
- `mortality`: μ, the yearly chance that a tree with symptoms dies of the disease.
- `inoculum`: c, the infected rachis it produces, relative to an ordinary tree.
- `offspring`: the class of the recruits inheriting from it. A class made by a treatment names
  the untreated class here, as a vaccinated tree's seedlings are not vaccinated.
"""
struct HostClass
    name::Symbol
    share::Float64
    susceptibility::Float64
    mortality::Float64
    inoculum::Float64
    offspring::Symbol
    function HostClass(name::Symbol; share::Real = 1.0, susceptibility::Real = 1.0,
                       mortality::Real = 0.0, inoculum::Real = 1.0, offspring::Symbol = name)
        0 <= share <= 1 || throw(ArgumentError("class $name: share $share is outside [0, 1]"))
        susceptibility >= 0 ||
            throw(ArgumentError("class $name: susceptibility $susceptibility is negative"))
        0 <= mortality <= 1 ||
            throw(ArgumentError("class $name: mortality $mortality is outside [0, 1]"))
        inoculum >= 0 || throw(ArgumentError("class $name: inoculum $inoculum is negative"))
        return new(name, share, susceptibility, mortality, inoculum, offspring)
    end
end

"""
    vaccinated_class(from::HostClass; efficacy, mortality = from.mortality,
                     inoculum = from.inoculum, name = Symbol(:vaccinated_, from.name))

The class that [`Vaccinate`](@ref) moves trees of `from` into. A generic vaccine, with no product
assumed: `efficacy` is ε, so the susceptibility is (1 − ε) s. It starts empty, and its recruits
belong to `from`.
"""
function vaccinated_class(from::HostClass; efficacy::Real, mortality::Real = from.mortality,
                          inoculum::Real = from.inoculum, name::Symbol = Symbol(:vaccinated_, from.name))
    0 <= efficacy <= 1 || throw(ArgumentError("efficacy $efficacy is outside [0, 1]"))
    return HostClass(name; share = 0.0, susceptibility = (1 - efficacy) * from.susceptibility,
                     mortality, inoculum, offspring = from.name)
end

"""
    HostDynamics(classes; background_mortality = 0.0, juvenile_years = 1.0,
                 juvenile_mortality_factor = 1.0, juvenile_inoculum = 0.0,
                 regeneration_rate = 0.0, heritability = 0.0)

How the ash itself changes from year to year, around the spore model of the paper. The host in
each quadrat is divided into `classes` ([`HostClass`](@ref)), each with adults and juveniles,
healthy or with symptoms. Each year, in this order:

1. Symptoms: with q̃ = F f(T)/r_S as in the paper, a tree at risk develops them with chance
   min(1, s q̃). Trees at risk are the healthy ones and the share 1 − C_pers that recover
   (C_pers from [`persistence`](@ref) when the settings' `heat_recovery` is not 0).
2. Death: trees with symptoms die of the disease with chance μ, multiplied by
   `juvenile_mortality_factor` for juveniles and capped at 1, then all trees die with chance
   `background_mortality`.
3. Maturation: a share 1/`juvenile_years` of surviving juveniles become adults.
4. Recruitment: healthy juveniles are added at a rate
   (A/A₀) m N + ρ A max(0, 1 − N/K), where A is adult area, N all live area and m the
   background mortality, all before this year's changes, ρ is `regeneration_rate`, and K and A₀
   are the starting live and adult area of the quadrat. The first term replaces background
   deaths, so an undiseased stand stays as it started, and the second fills gaps left by the
   disease. A share `heritability` of recruits follow their parents' classes, by adult area,
   and the rest follow each class's `share`.
5. Interventions (see [`AbstractIntervention`](@ref)).

Live trees produce infected rachis in proportion to c, or c times `juvenile_inoculum` for
juveniles, so that χ = F Σ c (area) replaces χ = F d of the paper. Juveniles start at the share
that an undiseased stand keeps. [`constant_host`](@ref) gives the paper's model.
"""
struct HostDynamics
    classes::Vector{HostClass}
    background_mortality::Float64
    juvenile_years::Float64
    juvenile_mortality_factor::Float64
    juvenile_inoculum::Float64
    regeneration_rate::Float64
    heritability::Float64
    function HostDynamics(classes::AbstractVector{HostClass}; background_mortality::Real = 0.0,
                          juvenile_years::Real = 1.0, juvenile_mortality_factor::Real = 1.0,
                          juvenile_inoculum::Real = 0.0, regeneration_rate::Real = 0.0,
                          heritability::Real = 0.0)
        isempty(classes) && throw(ArgumentError("a host needs at least one class"))
        total = sum(c.share for c in classes)
        isapprox(total, 1; atol = 1e-12) ||
            throw(ArgumentError("the classes' shares sum to $total, not 1"))
        names = [c.name for c in classes]
        allunique(names) || throw(ArgumentError("class names repeat: $names"))
        for c in classes
            c.offspring in names ||
                throw(ArgumentError("class $(c.name): offspring $(c.offspring) is not among $names"))
        end
        0 <= background_mortality < 1 ||
            throw(ArgumentError("background_mortality $background_mortality is outside [0, 1)"))
        juvenile_years >= 1 ||
            throw(ArgumentError("juvenile_years $juvenile_years is below 1"))
        juvenile_mortality_factor >= 0 ||
            throw(ArgumentError("juvenile_mortality_factor $juvenile_mortality_factor is negative"))
        juvenile_inoculum >= 0 ||
            throw(ArgumentError("juvenile_inoculum $juvenile_inoculum is negative"))
        regeneration_rate >= 0 ||
            throw(ArgumentError("regeneration_rate $regeneration_rate is negative"))
        0 <= heritability <= 1 ||
            throw(ArgumentError("heritability $heritability is outside [0, 1]"))
        return new(collect(classes), background_mortality, juvenile_years,
                   juvenile_mortality_factor, juvenile_inoculum, regeneration_rate, heritability)
    end
end

"""
    constant_host()

The host of the paper: one ordinary class, with no deaths or recruits, so that
`solve(HostProblem(prob, constant_host()))` reproduces `solve(prob)`.
"""
constant_host() = HostDynamics([HostClass(:ash)])

"""
    AbstractIntervention

Supertype of what can be done to the host: [`Vaccinate`](@ref), [`PlantStock`](@ref),
[`Fell`](@ref) and [`ReduceLitter`](@ref). Interventions apply in a [`HostProblem`](@ref) after the
year's changes to the host and before its rachis is counted, in the order given. Each has
`years`, when it acts, and `cells`, a Bool matrix of the quadrats it acts on (`nothing` for all),
which is how targeting by region or host density is expressed. The area each one acts on per year
is in `HostSolution.applied`.
"""
abstract type AbstractIntervention end

const CellMask = Union{Nothing,BitMatrix}
to_mask(cells::Nothing) = nothing
to_mask(cells::AbstractMatrix{Bool}) = BitMatrix(cells)
in_cells(cells::Nothing, i, j) = true
in_cells(cells::BitMatrix, i, j) = cells[i, j]

function check_rate(name, value)
    0 <= value <= 1 || throw(ArgumentError("$name $value is outside [0, 1]"))
    return Float64(value)
end

"""
    Vaccinate(; from, to, years, rate, max_coverage = 1.0, waning = 0.0, juveniles = true,
              symptomatic = false, cells = nothing)

Each year in `years`, move a share `rate` of the eligible trees of class `from` to class `to`
(usually a [`vaccinated_class`](@ref)), until `to` holds a share `max_coverage` of the two
together. Healthy trees are eligible, with juveniles if `juveniles`, and trees with symptoms if
`symptomatic`, which stay symptomatic but take on the mortality of `to`.

Protection is lost at the yearly rate `waning` in every year, including after `years`: a share of
`to` returns to `from`, in whatever state it is. Repeat doses need no separate setting, since
waned trees are eligible again and count towards the area treated.
"""
struct Vaccinate <: AbstractIntervention
    from::Symbol
    to::Symbol
    years::UnitRange{Int}
    rate::Float64
    max_coverage::Float64
    waning::Float64
    juveniles::Bool
    symptomatic::Bool
    cells::CellMask
end
function Vaccinate(; from::Symbol, to::Symbol, years::AbstractUnitRange{<:Integer}, rate::Real,
                   max_coverage::Real = 1.0, waning::Real = 0.0, juveniles::Bool = true,
                   symptomatic::Bool = false, cells = nothing)
    from == to && throw(ArgumentError("Vaccinate: from and to are both $from"))
    return Vaccinate(from, to, years, check_rate("rate", rate), check_rate("max_coverage", max_coverage),
                     check_rate("waning", waning), juveniles, symptomatic, to_mask(cells))
end

"""
    PlantStock(; class, years, rate, cells = nothing)

Each year in `years`, plant healthy juveniles of `class`, up to a share `rate` of the quadrat's
starting host, without taking live area above that starting host. So planting fills gaps, and its
lag before it matters is `juvenile_years` of the [`HostDynamics`](@ref).
"""
struct PlantStock <: AbstractIntervention
    class::Symbol
    years::UnitRange{Int}
    rate::Float64
    cells::CellMask
end
function PlantStock(; class::Symbol, years::AbstractUnitRange{<:Integer}, rate::Real, cells = nothing)
    return PlantStock(class, years, check_rate("rate", rate), to_mask(cells))
end

"""
    Fell(; years, rate, cells = nothing)

Each year in `years`, remove a share `rate` of adults with symptoms in every class. Felled area is
not counted in `disease_deaths`.
"""
struct Fell <: AbstractIntervention
    years::UnitRange{Int}
    rate::Float64
    cells::CellMask
end
function Fell(; years::AbstractUnitRange{<:Integer}, rate::Real, cells = nothing)
    return Fell(years, check_rate("rate", rate), to_mask(cells))
end

"""
    ReduceLitter(; years, effect, cells = nothing)

Each year in `years`, remove a share `effect` of the year's infected rachis, by clearing or
treating leaf litter. It lowers χ of that year, so the spores of the next two years. It does not
act on the trees, or on the spores that arrive from elsewhere.
"""
struct ReduceLitter <: AbstractIntervention
    years::UnitRange{Int}
    effect::Float64
    cells::CellMask
end
function ReduceLitter(; years::AbstractUnitRange{<:Integer}, effect::Real, cells = nothing)
    return ReduceLitter(years, check_rate("effect", effect), to_mask(cells))
end

referenced_classes(::AbstractIntervention) = Symbol[]
referenced_classes(v::Vaccinate) = [v.from, v.to]
referenced_classes(p::PlantStock) = [p.class]

"""
    HostProblem(prob::AshDiebackProblem, host::HostDynamics; interventions = [])

The model `prob` with the changing host `host` in place of its constant ash density, and any
[`AbstractIntervention`](@ref)s. Solve it with `solve`, which gives a [`HostSolution`](@ref). The
data of `prob` supply the starting host (`ash_density`), the climate and the imputed first two
years, so a future climate is given by building `prob` with a longer climate array and a later
`final_year`.
"""
struct HostProblem{P<:AshDiebackProblem}
    prob::P
    host::HostDynamics
    interventions::Vector{AbstractIntervention}
    function HostProblem(prob::P, host::HostDynamics;
                         interventions::AbstractVector = AbstractIntervention[]) where {P<:AshDiebackProblem}
        names = [c.name for c in host.classes]
        for iv in interventions
            for c in referenced_classes(iv)
                c in names || throw(ArgumentError("$(typeof(iv)): class $c is not among $names"))
            end
            cells = iv.cells
            (cells === nothing || size(cells) == size(prob.data.ash_density)) ||
                throw(ArgumentError("$(typeof(iv)): cells have size $(size(cells)), the grid " *
                                    "$(size(prob.data.ash_density))"))
        end
        return new{P}(prob, host, collect(AbstractIntervention, interventions))
    end
end

"""
    HostSolution

The result of `solve(::HostProblem)`.

Fields, with the host in the units of `ash_density` and years given by [`host_years`](@ref):
- `fungi`: colonised rachis per ash F, size `(nx, ny, years - 1)`, for the
  [`simulated_years`](@ref). F of year a drives the symptoms of year a + 1.
- `spore_load`: the diffused spore load w before the cap, F = [`saturate`](@ref)(w, S) (or the
  power law of `spore_exponent` in [`ModelSettings`](@ref)), of the same size; `NaN` in the two imputed years. w / S is how far a quadrat is above the cap.
- `healthy`, `symptomatic`: adults, size `(nx, ny, classes, years)`.
- `juvenile_healthy`, `juvenile_symptomatic`: juveniles, of the same size.
- `disease_deaths`: cumulative area killed by the disease, size `(nx, ny, years)`.
- `applied`: the area each intervention acted on, size `(interventions, years)`: trees vaccinated,
  planted or felled, or adults in quadrats whose litter was treated.
- `prob`, `alg`: the [`HostProblem`](@ref) and the diffusion algorithm.
"""
struct HostSolution{P<:HostProblem,A<:AbstractDiffusionAlgorithm}
    fungi::Array{Float64,3}
    spore_load::Array{Float64,3}
    healthy::Array{Float64,4}
    symptomatic::Array{Float64,4}
    juvenile_healthy::Array{Float64,4}
    juvenile_symptomatic::Array{Float64,4}
    disease_deaths::Array{Float64,3}
    applied::Matrix{Float64}
    prob::P
    alg::A
end

"""
    host_years(sol::HostSolution)

The years of the host arrays: the first imputed year (with no symptoms), then the survey years.
"""
function host_years(sol::HostSolution)
    s = sol.prob.prob.settings
    return s.initialization_year:s.final_year
end

"""
    live_host(sol::HostSolution; juveniles = true)

Live host area in each quadrat and year, size `(nx, ny, years)`, summed over classes.
"""
function live_host(sol::HostSolution; juveniles::Bool = true)
    live = dropdims(sum(sol.healthy .+ sol.symptomatic; dims = 3); dims = 3)
    juveniles &&
        (live .+= dropdims(sum(sol.juvenile_healthy .+ sol.juvenile_symptomatic; dims = 3); dims = 3))
    return live
end

"""
    symptom_share(sol::HostSolution)

The share of live adults with symptoms in each quadrat and year, comparable with the paper's
q when the host is constant. It is 0 where no adults are alive.
"""
function symptom_share(sol::HostSolution)
    Y = dropdims(sum(sol.symptomatic; dims = 3); dims = 3)
    A = Y .+ dropdims(sum(sol.healthy; dims = 3); dims = 3)
    return @. ifelse(A > 0, Y / A, 0.0)
end

class_index(classes, name) = findfirst(c -> c.name == name, classes)

function apply_intervention!(v::Vaccinate, st, litter, classes, K, l, year)
    H, Y, JH, JY = st
    a, b = class_index(classes, v.from), class_index(classes, v.to)
    nx, ny = size(K)
    active = year in v.years
    area = 0.0
    for j in 1:ny, i in 1:nx
        K[i, j] > 0 || continue
        if v.waning > 0
            for A in st
                moved = v.waning * A[i, j, b, l]
                A[i, j, b, l] -= moved
                A[i, j, a, l] += moved
            end
        end
        (active && in_cells(v.cells, i, j)) || continue
        eligible = H[i, j, a, l] + (v.juveniles ? JH[i, j, a, l] : 0.0) +
                   (v.symptomatic ? Y[i, j, a, l] + (v.juveniles ? JY[i, j, a, l] : 0.0) : 0.0)
        eligible > 0 || continue
        treated = H[i, j, b, l] + Y[i, j, b, l] + JH[i, j, b, l] + JY[i, j, b, l]
        untreated = H[i, j, a, l] + Y[i, j, a, l] + JH[i, j, a, l] + JY[i, j, a, l]
        room = max(0.0, v.max_coverage * (treated + untreated) - treated)
        dose = min(v.rate * eligible, room)
        share = dose / eligible
        area += dose
        moved = share * H[i, j, a, l]
        H[i, j, a, l] -= moved
        H[i, j, b, l] += moved
        if v.juveniles
            moved = share * JH[i, j, a, l]
            JH[i, j, a, l] -= moved
            JH[i, j, b, l] += moved
        end
        if v.symptomatic
            moved = share * Y[i, j, a, l]
            Y[i, j, a, l] -= moved
            Y[i, j, b, l] += moved
            if v.juveniles
                moved = share * JY[i, j, a, l]
                JY[i, j, a, l] -= moved
                JY[i, j, b, l] += moved
            end
        end
    end
    return area
end

function apply_intervention!(p::PlantStock, st, litter, classes, K, l, year)
    year in p.years || return 0.0
    H, Y, JH, JY = st
    k = class_index(classes, p.class)
    nx, ny = size(K)
    area = 0.0
    for j in 1:ny, i in 1:nx
        (K[i, j] > 0 && in_cells(p.cells, i, j)) || continue
        live = 0.0
        for c in eachindex(classes)
            live += H[i, j, c, l] + Y[i, j, c, l] + JH[i, j, c, l] + JY[i, j, c, l]
        end
        planted = min(p.rate * K[i, j], max(0.0, K[i, j] - live))
        JH[i, j, k, l] += planted
        area += planted
    end
    return area
end

function apply_intervention!(f::Fell, st, litter, classes, K, l, year)
    year in f.years || return 0.0
    _, Y, _, _ = st
    nx, ny = size(K)
    area = 0.0
    for c in eachindex(classes), j in 1:ny, i in 1:nx
        in_cells(f.cells, i, j) || continue
        felled = f.rate * Y[i, j, c, l]
        Y[i, j, c, l] -= felled
        area += felled
    end
    return area
end

function apply_intervention!(r::ReduceLitter, st, litter, classes, K, l, year)
    year in r.years || return 0.0
    H, Y, _, _ = st
    nx, ny = size(K)
    area = 0.0
    for j in 1:ny, i in 1:nx
        in_cells(r.cells, i, j) || continue
        litter[i, j] *= 1 - r.effect
        for c in eachindex(classes)
            area += H[i, j, c, l] + Y[i, j, c, l]
        end
    end
    return area
end

"""
    solve(prob::HostProblem, alg = SparseCrankNicolson())

Simulate the spore model and the host together. The spore source of year a becomes
β (χ_{a−2} + χ_{a−1}) with χ = F × (live host weighted by its inoculum), then diffusion and the cap
at S as in the paper.
"""
function CommonSolve.solve(hp::HostProblem, alg::AbstractDiffusionAlgorithm = SparseCrankNicolson())
    prob, host = hp.prob, hp.host
    θ, d, s = prob.p, prob.data, prob.settings
    nx, ny = size(d.grid)
    classes = host.classes
    nk = length(classes)
    sim = simulated_years(s)
    n = length(sim)
    parent = [class_index(classes, c.offspring) for c in classes]

    F = zeros(nx, ny, n)
    F[:, :, 1], F[:, :, 2] = initial_fungi(prob)
    W = fill(NaN, nx, ny, n)
    H, Y = zeros(nx, ny, nk, n + 1), zeros(nx, ny, nk, n + 1)
    JH, JY = zeros(nx, ny, nk, n + 1), zeros(nx, ny, nk, n + 1)
    dead = zeros(nx, ny, n + 1)
    applied = zeros(length(hp.interventions), n + 1)

    m = host.background_mortality
    g = 1 / host.juvenile_years
    j0 = m == 0 ? 0.0 : m / (1 - (1 - m) * (1 - g))
    K = d.ash_density
    for k in 1:nk, j in 1:ny, i in 1:nx
        H[i, j, k, 1] = classes[k].share * K[i, j] * (1 - j0)
        JH[i, j, k, 1] = classes[k].share * K[i, j] * j0
    end

    # Interventions, then the weighted host of the year: χ = F × D
    D = zeros(nx, ny, n + 1)
    litter = ones(nx, ny)
    state = (H, Y, JH, JY)
    function close_year!(l)
        fill!(litter, 1.0)
        for (q, iv) in enumerate(hp.interventions)
            applied[q, l] = apply_intervention!(iv, state, litter, classes, K, l, first(sim) + l - 1)
        end
        for k in 1:nk, j in 1:ny, i in 1:nx
            D[i, j, l] += classes[k].inoculum * (H[i, j, k, l] + Y[i, j, k, l] +
                                                 host.juvenile_inoculum *
                                                 (JH[i, j, k, l] + JY[i, j, k, l]))
        end
        view(D, :, :, l) .*= litter
    end
    close_year!(1)

    op = diffusion_operator(alg, prob.laplacian, θ.D, prob.scheme)
    t0 = first(sim) - first(d.climate_years) + 1
    C = θ.C_pers
    for l in 1:n
        if l >= 3
            spores = view(F, :, :, l - 2) .* view(D, :, :, l - 2) .+
                     view(F, :, :, l - 1) .* view(D, :, :, l - 1)
            if θ.r_allee != 0
                @. spores = ifelse(spores < θ.r_allee, spores * (spores / θ.r_allee), spores)
            end
            if θ.beta_1 != 0
                spores .*= θ.beta_0 .+ θ.beta_1 .* view(d.rainfall, :, :, t0 + l - 1)
            else
                spores .*= θ.beta_0
            end
            w = diffuse(op, vec(spores), prob.scheme)
            W[:, :, l] .= reshape(w, nx, ny)
            F[:, :, l] .= rachis_response.(view(W, :, :, l), θ.S_sat, Ref(s))
        end
        dead[:, :, l + 1] .= view(dead, :, :, l)
        for j in 1:ny, i in 1:nx
            K[i, j] > 0 || continue
            qt = F[i, j, l] * temperature_effect(d.temperature[i, j, t0 + l - 1], θ.gamma, θ.kappa) /
                 θ.r_symp
            Ce = persistence(C, d.temperature[i, j, t0 + l - s.heat_recovery_lag], θ.gamma, s.heat_recovery)
            A = N = 0.0
            for k in 1:nk
                adults = H[i, j, k, l] + Y[i, j, k, l]
                A += adults
                N += adults + JH[i, j, k, l] + JY[i, j, k, l]
            end
            for k in 1:nk
                c = classes[k]
                p = min(1.0, c.susceptibility * qt)
                μj = min(1.0, c.mortality * host.juvenile_mortality_factor)
                # Symptoms, as q = q̃ (1 − C q) + C q of the paper
                risk = H[i, j, k, l] + (1 - Ce) * Y[i, j, k, l]
                ya = Ce * Y[i, j, k, l] + p * risk
                ha = (1 - p) * risk
                risk = JH[i, j, k, l] + (1 - Ce) * JY[i, j, k, l]
                yj = Ce * JY[i, j, k, l] + p * risk
                hj = (1 - p) * risk
                # Death
                dead[i, j, l + 1] += c.mortality * ya + μj * yj
                ya *= (1 - c.mortality) * (1 - m)
                ha *= 1 - m
                yj *= (1 - μj) * (1 - m)
                hj *= 1 - m
                # Maturation
                H[i, j, k, l + 1] = ha + g * hj
                Y[i, j, k, l + 1] = ya + g * yj
                JH[i, j, k, l + 1] = (1 - g) * hj
                JY[i, j, k, l + 1] = (1 - g) * yj
            end
            # Recruitment
            A > 0 || continue
            A0 = K[i, j] * (1 - j0)
            births = A / A0 * m * N + host.regeneration_rate * A * max(0.0, 1 - N / K[i, j])
            births > 0 || continue
            for k in 1:nk
                inherited = (H[i, j, k, l] + Y[i, j, k, l]) / A
                JH[i, j, parent[k], l + 1] += births * host.heritability * inherited
                JH[i, j, k, l + 1] += births * (1 - host.heritability) * classes[k].share
            end
        end
        close_year!(l + 1)
    end
    return HostSolution(F, W, H, Y, JH, JY, dead, applied, hp, alg)
end
