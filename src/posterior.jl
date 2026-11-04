"""
    peoc_prior_ranges()

The uniform prior ranges `[lower upper]` of the ten parameters shipped with PEOC 1.0
(`1_or_2_parameters_of_the_AMIS_or_MH_algorithm.py`), one row per parameter in the order of
[`AshDiebackParameters`](@ref). A row with `lower == upper` fixes that parameter
(`r_symp = 1000`).
"""
peoc_prior_ranges() = [1000.0 1000.0
                       0.0 5000.0
                       0.0 800.0
                       0.0 5000.0
                       0.0 200.0
                       0.0 500.0
                       0.0 600.0
                       0.0 1.0
                       0.0 200.0
                       0.0 10.0]

"""
    reduced_model_prior_ranges()

Prior ranges of the paper's reduced model (eq. 13), used for its AMIS results: no Allee
effect and no rainfall effect (`r_allee = beta_1 = 0`), `gamma = 21.50001`, `C_pers = 1`
and `r_symp = 1000` fixed; `beta_0`, `D`, `S_sat`, `C_init` and `kappa` free with the ranges
of [`peoc_prior_ranges`](@ref).
"""
function reduced_model_prior_ranges()
    R = peoc_prior_ranges()
    R[2, :] .= 0.0
    R[4, :] .= 0.0
    R[7, :] .= 21.50001
    R[8, :] .= 1.0
    return R
end

"""
    AshDiebackPosterior(prob::AshDiebackProblem, ranges = peoc_prior_ranges();
                        alg = SparseCrankNicolson())

The posterior of the model parameters under independent uniform priors on `ranges` (a
`10 × 2` matrix of `[lower upper]` rows; equal bounds fix a parameter), as a
LogDensityProblems.jl log density over the free parameters only.

Use [`free_parameters`](@ref) to see which parameters are free and [`full_parameters`](@ref)
to turn a point of the posterior back into an [`AshDiebackParameters`](@ref). The log
density is `-Inf` outside the prior and where the likelihood is zero, and is evaluated with
the diffusion algorithm `alg`.
"""
struct AshDiebackPosterior{P<:AshDiebackProblem,A<:AbstractDiffusionAlgorithm}
    problem::P
    ranges::Matrix{Float64}
    free::Vector{Int}
    alg::A
end

function AshDiebackPosterior(prob::AshDiebackProblem, ranges::AbstractMatrix = peoc_prior_ranges();
                             alg::AbstractDiffusionAlgorithm = SparseCrankNicolson())
    size(ranges) == (10, 2) ||
        throw(ArgumentError("ranges must be a 10 × 2 matrix, not $(size(ranges))"))
    all(ranges[:, 1] .<= ranges[:, 2]) ||
        throw(ArgumentError("every lower bound must be at most its upper bound"))
    free = findall(ranges[:, 1] .< ranges[:, 2])
    return AshDiebackPosterior(prob, Matrix{Float64}(ranges), free, alg)
end

"""
    free_parameters(post::AshDiebackPosterior)

The names of the free parameters of `post`, in the order of its log density's argument.
"""
free_parameters(post::AshDiebackPosterior) = collect(parameter_names()[post.free])

"""
    full_parameters(post::AshDiebackPosterior, x::AbstractVector)

The [`AshDiebackParameters`](@ref) with the free parameters set to `x` and the fixed ones to
their prior value.
"""
function full_parameters(post::AshDiebackPosterior, x::AbstractVector)
    length(x) == length(post.free) ||
        throw(ArgumentError("expected $(length(post.free)) free parameters, got $(length(x))"))
    v = Vector{Float64}(post.ranges[:, 1])
    v[post.free] .= x
    return AshDiebackParameters(v)
end

"""
    logprior(post::AshDiebackPosterior, x::AbstractVector)

Log density of the uniform prior at the free parameters `x`: `-Inf` outside the ranges.
"""
function logprior(post::AshDiebackPosterior, x::AbstractVector)
    lp = 0.0
    for (k, j) in enumerate(post.free)
        lo, hi = post.ranges[j, 1], post.ranges[j, 2]
        (lo <= x[k] <= hi) || return -Inf
        lp -= log(hi - lo)
    end
    return lp
end

"""
    loglikelihood(post::AshDiebackPosterior, x::AbstractVector)

The log-likelihood at the free parameters `x` (not checked against the prior).
"""
function loglikelihood(post::AshDiebackPosterior, x::AbstractVector)
    return loglikelihood(remake(post.problem; p = full_parameters(post, x)), post.alg)
end

function LogDensityProblems.logdensity(post::AshDiebackPosterior, x::AbstractVector)
    lp = logprior(post, x)
    isfinite(lp) || return -Inf
    return lp + loglikelihood(post, x)
end

LogDensityProblems.dimension(post::AshDiebackPosterior) = length(post.free)

function LogDensityProblems.capabilities(::Type{<:AshDiebackPosterior})
    return LogDensityProblems.LogDensityOrder{0}()
end
