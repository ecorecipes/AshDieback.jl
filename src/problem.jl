"""
    ModelSettings(; initialization_year = 2007, final_year = 2019, observed_trees = 30,
                  delta_t = 0.1, diffusion_period = 60.0, cap_symptom_probability = false)

Settings of the model that are not estimated (PEOC's `parameters_of_the_model.py`).

- `initialization_year`: the first of the two years whose colonised rachis is imputed;
  diffusion is simulated from `initialization_year + 2`.
- `final_year`: the last survey year in the likelihood (2019 in the paper; later years are
  kept for validation).
- `observed_trees`: trees observed per plot visit (m = 30).
- `delta_t`, `diffusion_period`: time step and duration τ (days) of the spore diffusion.
- `cap_symptom_probability`: cap the yearly probability of developing symptoms at 1, as in
  eq. (6) of the paper. PEOC does not, which gives a `NaN` likelihood when it exceeds 1;
  `false` reproduces PEOC.
"""
Base.@kwdef struct ModelSettings
    initialization_year::Int = 2007
    final_year::Int = 2019
    observed_trees::Int = 30
    delta_t::Float64 = 0.1
    diffusion_period::Float64 = 60.0
    cap_symptom_probability::Bool = false
end

"""
    simulated_years(settings::ModelSettings)

The years whose colonised rachis per ash F_a = χ_a/d the model holds: the two imputed years
and the simulated ones (2007:2018 by default). F_{a−1} drives the symptoms of year a.
"""
simulated_years(s::ModelSettings) = s.initialization_year:(s.final_year - 1)

"""
    survey_years(settings::ModelSettings)

The survey years that enter the likelihood (2008:2019 by default).
"""
survey_years(s::ModelSettings) = (s.initialization_year + 1):s.final_year

struct CrankNicolsonScheme
    h::Float64
    delta_t::Float64
    diffusion_period::Float64
    nsteps::Int
end

nsteps(s::CrankNicolsonScheme) = s.nsteps

function CrankNicolsonScheme(grid::QuadratGrid, s::ModelSettings)
    # int(60/0.1) in PEOC
    return CrankNicolsonScheme(grid.quadrat_length, s.delta_t, s.diffusion_period,
                               trunc(Int, s.diffusion_period / s.delta_t))
end

"""
    BinomialTerms

Constants of the binomial likelihood of one survey year: for each visited quadrat, the
number of trees with symptoms `k = round(n p)` (ties to even) and without, `n − k`, out of
`n = m × visits`, and `log_choose = log C(n, k)`.
"""
struct BinomialTerms
    log_choose::Vector{Float64}
    infected::Vector{Float64}
    healthy::Vector{Float64}
end

function BinomialTerms(o::YearObservations, observed_trees::Integer)
    n = observed_trees .* Float64.(o.visits)
    k = Float64.(round.(Int, n .* o.proportion))
    nk = n .- k
    return BinomialTerms(loggamma.(n .+ 1) .- (loggamma.(k .+ 1) .+ loggamma.(nk .+ 1)),
                         k, nk)
end

"""
    AshDiebackProblem(data::PEOCData, p; settings = ModelSettings())

The mechanistic-statistical ash dieback model of Fritsch et al. on `data`, with parameters
`p` (an [`AshDiebackParameters`](@ref) or a vector in its order).

`solve(prob)` simulates the colonised rachis and the probability of symptoms in every
quadrat (an [`AshDiebackSolution`](@ref)); `loglikelihood(sol)` scores it against the
survey. `remake(prob; p = θ)` changes the parameters and reuses everything else.
"""
struct AshDiebackProblem{P<:AshDiebackParameters}
    data::PEOCData
    p::P
    settings::ModelSettings
    scheme::CrankNicolsonScheme
    laplacian::SparseMatrixCSC{Float64,Int}
    binomial::Vector{BinomialTerms}
    survey::Vector{YearObservations}
end

function AshDiebackProblem(data::PEOCData, p::AbstractVector;
                           settings::ModelSettings = ModelSettings())
    years = survey_years(settings)
    obs_years = [o.year for o in data.observations]
    all(in(obs_years), years) ||
        throw(ArgumentError("the survey has years $(first(obs_years)):$(last(obs_years)); " *
                            "the likelihood needs $years"))
    sim = simulated_years(settings)
    (first(sim) in data.climate_years && last(sim) in data.climate_years) ||
        throw(ArgumentError("the climate data covers $(data.climate_years); the model " *
                            "needs $sim"))
    survey = [data.observations[findfirst(==(a), obs_years)] for a in years]
    binomial = [BinomialTerms(o, settings.observed_trees) for o in survey]
    return AshDiebackProblem(data, AshDiebackParameters(p), settings,
                             CrankNicolsonScheme(data.grid, settings),
                             neumann_laplacian(size(data.grid)...), binomial, survey)
end

"""
    remake(prob::AshDiebackProblem; p = prob.p)

A copy of `prob` with parameters `p`; the data, settings and precomputed constants are
shared.
"""
function SciMLBase.remake(prob::AshDiebackProblem; p = prob.p)
    return AshDiebackProblem(prob.data, AshDiebackParameters(p), prob.settings, prob.scheme,
                             prob.laplacian, prob.binomial, prob.survey)
end

function Base.show(io::IO, ::MIME"text/plain", prob::AshDiebackProblem)
    s = prob.settings
    println(io, "AshDiebackProblem: survey years ", first(survey_years(s)), ":",
            last(survey_years(s)), ", ", prob.data.temperature_index, " temperature index")
    print(io, "  p = ", NamedTuple{parameter_names()}(Tuple(prob.p)))
end

"""
    AshDiebackSolution

The simulated epidemic, returned by `solve(prob::AshDiebackProblem)`.

Fields:
- `fungi::Array{Float64,3}`: colonised rachis per ash F_a = χ_a/d, size `(nx, ny, years)`,
  for the years [`simulated_years`](@ref) (the first two imputed).
- `symptoms::Array{Float64,3}`: probability q_a that a tree has symptoms, for the
  [`survey_years`](@ref).
- `prob`, `alg`: the problem and the diffusion algorithm.
"""
struct AshDiebackSolution{P<:AshDiebackProblem,A<:AbstractDiffusionAlgorithm}
    fungi::Array{Float64,3}
    symptoms::Array{Float64,3}
    prob::P
    alg::A
end

"""
    initial_fungi(prob::AshDiebackProblem)

The imputed colonised rachis per ash of the two years before the first simulated one,
obtained by inverting the symptom model with q ≈ C_init Ψ (PEOC's `relative_fongi_1/2`).
Where f(T) = 0 it is replaced by 1 before dividing.
"""
function initial_fungi(prob::AshDiebackProblem)
    θ, d = prob.p, prob.data
    a0 = prob.settings.initialization_year - first(d.climate_years) + 1
    f(T) = (v = temperature_effect(T, θ.gamma, θ.kappa); v == 0 ? 1.0 : v)
    f1 = f.(view(d.temperature, :, :, a0))
    f2 = f.(view(d.temperature, :, :, a0 + 1))
    psi1, psi2 = d.psi
    C, Ci, rS = θ.C_pers, θ.C_init, θ.r_symp
    F1 = Ci * rS .* psi1 ./ f1
    F2 = Ci * rS .* ((psi2 .- C .* psi1) ./ ((1 .- C .* psi1 .* Ci) .* f2))
    return F1, F2
end

"""
    solve(prob::AshDiebackProblem, alg = SparseCrankNicolson())

Simulate the model: impute the first two years, then for each later year produce spores
from the rachis of the two previous years (with the Allee effect and rainfall), diffuse
them with `alg` (an [`AbstractDiffusionAlgorithm`](@ref)) and saturate them at `S_sat`;
finally compute the probability of symptoms for each survey year.
"""
function CommonSolve.solve(prob::AshDiebackProblem,
                           alg::AbstractDiffusionAlgorithm = SparseCrankNicolson())
    θ, d, s = prob.p, prob.data, prob.settings
    nx, ny = size(d.grid)
    years = simulated_years(s)
    F = zeros(nx, ny, length(years))
    F[:, :, 1], F[:, :, 2] = initial_fungi(prob)
    op = diffusion_operator(alg, prob.laplacian, θ.D, prob.scheme)
    rain0 = first(years) - first(d.climate_years) + 1
    for a in 3:length(years)
        spores = (F[:, :, a - 2] .+ F[:, :, a - 1]) .* d.ash_density
        if θ.r_allee != 0
            @. spores = ifelse(spores < θ.r_allee, spores * (spores / θ.r_allee), spores)
        end
        if θ.beta_1 != 0
            spores .*= θ.beta_0 .+ θ.beta_1 .* view(d.rainfall, :, :, rain0 + a - 1)
        else
            spores .*= θ.beta_0
        end
        w = diffuse(op, vec(spores), prob.scheme)
        F[:, :, a] .= min.(reshape(w, nx, ny), θ.S_sat)
    end
    return AshDiebackSolution(F, symptom_probability(prob, F), prob, alg)
end

"""
    symptom_probability(prob::AshDiebackProblem, fungi)

The probability q_a that a tree has symptoms in each survey year, from the colonised rachis
per ash `fungi` of the year before: q̃_a = F_{a−1} f(T_{a−1}) / r_S of developing them, and
q_a = q̃_a (1 − C_pers q_{a−1}) + C_pers q_{a−1} with q = 0 before the first survey year.
"""
function symptom_probability(prob::AshDiebackProblem, F::AbstractArray{<:Real,3})
    θ, d, s = prob.p, prob.data, prob.settings
    nx, ny = size(d.grid)
    t0 = s.initialization_year - first(d.climate_years) + 1
    n = length(survey_years(s))
    Q = zeros(nx, ny, n)
    q = zeros(nx, ny)
    C = θ.C_pers
    for l in 1:n
        T = view(d.temperature, :, :, t0 + l - 1)
        qt = view(F, :, :, l) .* temperature_effect.(T, θ.gamma, θ.kappa) ./ θ.r_symp
        s.cap_symptom_probability && (qt .= min.(qt, 1))
        q = qt .* (1 .- C .* q) .+ C .* q
        Q[:, :, l] .= q
    end
    return Q
end
