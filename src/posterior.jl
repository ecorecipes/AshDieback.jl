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
                        alg = SparseCrankNicolson(), cap_exponent = nothing)

The posterior of the model parameters under independent uniform priors on `ranges` (a
`10 × 2` matrix of `[lower upper]` rows; equal bounds fix a parameter), as a
LogDensityProblems.jl log density over the free parameters only.

`cap_exponent = (lo, hi)`, with 0 ≤ lo < hi ≤ 1, also frees the exponent b of the broken
power law above the spore cap (see [`ModelSettings`](@ref)) under a uniform prior on
`[lo, hi]`; it is then the last free parameter, after those of `ranges`. With `nothing`, the
default, b stays at `prob.settings.cap_exponent`.

Use [`free_parameters`](@ref) to see which parameters are free, [`full_parameters`](@ref)
to turn a point of the posterior back into an [`AshDiebackParameters`](@ref) and
[`posterior_problem`](@ref) for the problem at a point, exponent included. The log density
is `-Inf` outside the prior and where the likelihood is zero, and is evaluated with the
diffusion algorithm `alg`.
"""
struct AshDiebackPosterior{P<:AshDiebackProblem,A<:AbstractDiffusionAlgorithm}
    problem::P
    ranges::Matrix{Float64}
    free::Vector{Int}
    alg::A
    cap_exponent::Union{Nothing,Tuple{Float64,Float64}}
end

function AshDiebackPosterior(prob::AshDiebackProblem, ranges::AbstractMatrix = peoc_prior_ranges();
                             alg::AbstractDiffusionAlgorithm = SparseCrankNicolson(),
                             cap_exponent = nothing)
    size(ranges) == (10, 2) ||
        throw(ArgumentError("ranges must be a 10 × 2 matrix, not $(size(ranges))"))
    all(ranges[:, 1] .<= ranges[:, 2]) ||
        throw(ArgumentError("every lower bound must be at most its upper bound"))
    free = findall(ranges[:, 1] .< ranges[:, 2])
    if cap_exponent !== nothing
        length(cap_exponent) == 2 && 0 <= cap_exponent[1] < cap_exponent[2] <= 1 ||
            throw(ArgumentError("cap_exponent must be nothing or (lo, hi) with 0 ≤ lo < hi ≤ 1, got $cap_exponent"))
        s = prob.settings
        isinf(s.saturation_sharpness) && s.spore_exponent === nothing ||
            throw(ArgumentError("a free cap_exponent needs saturation_sharpness = Inf and spore_exponent = nothing"))
        cap_exponent = (Float64(cap_exponent[1]), Float64(cap_exponent[2]))
    end
    return AshDiebackPosterior(prob, Matrix{Float64}(ranges), free, alg, cap_exponent)
end

free_cap_exponent(post::AshDiebackPosterior) = post.cap_exponent !== nothing

"""
    free_parameters(post::AshDiebackPosterior)

The names of the free parameters of `post`, in the order of its log density's argument,
ending with `:cap_exponent` when the exponent is free.
"""
function free_parameters(post::AshDiebackPosterior)
    names = collect(parameter_names()[post.free])
    return free_cap_exponent(post) ? push!(names, :cap_exponent) : names
end

"""
    parameter_bounds(post::AshDiebackPosterior)

The lower and upper bounds of the uniform prior on the free parameters, as two vectors in
the order of [`free_parameters`](@ref).
"""
function parameter_bounds(post::AshDiebackPosterior)
    lo, hi = post.ranges[post.free, 1], post.ranges[post.free, 2]
    free_cap_exponent(post) && (push!(lo, post.cap_exponent[1]); push!(hi, post.cap_exponent[2]))
    return lo, hi
end

function check_dimension(post::AshDiebackPosterior, x::AbstractVector)
    d = LogDensityProblems.dimension(post)
    length(x) == d || throw(ArgumentError("expected $d free parameters, got $(length(x))"))
    return nothing
end

"""
    full_parameters(post::AshDiebackPosterior, x::AbstractVector)

The [`AshDiebackParameters`](@ref) with the free parameters set to `x` and the fixed ones to
their prior value. A free cap exponent, `x[end]`, is not a model parameter: it is in the
settings of [`posterior_problem`](@ref).
"""
function full_parameters(post::AshDiebackPosterior, x::AbstractVector)
    check_dimension(post, x)
    v = Vector{Float64}(post.ranges[:, 1])
    v[post.free] .= view(x, 1:length(post.free))
    return AshDiebackParameters(v)
end

"""
    posterior_problem(post::AshDiebackPosterior, x::AbstractVector)

The [`AshDiebackProblem`](@ref) at the free parameters `x`: `post.problem` with
[`full_parameters`](@ref)`(post, x)` and, when the cap exponent is free, `cap_exponent =
x[end]`. It shares the problem's precomputed constants.
"""
function posterior_problem(post::AshDiebackPosterior, x::AbstractVector)
    p = full_parameters(post, x)
    free_cap_exponent(post) || return remake(post.problem; p)
    return remake(post.problem; p, settings = with_cap_exponent(post.problem.settings, x[end]))
end

"""
    logprior(post::AshDiebackPosterior, x::AbstractVector)

Log density of the uniform prior at the free parameters `x`: `-Inf` outside the ranges.
"""
function logprior(post::AshDiebackPosterior, x::AbstractVector)
    check_dimension(post, x)
    lo, hi = parameter_bounds(post)
    lp = 0.0
    for k in eachindex(lo)
        (lo[k] <= x[k] <= hi[k]) || return -Inf
        lp -= log(hi[k] - lo[k])
    end
    return lp
end

"""
    loglikelihood(post::AshDiebackPosterior, x::AbstractVector)

The log-likelihood at the free parameters `x` (not checked against the prior).
"""
function loglikelihood(post::AshDiebackPosterior, x::AbstractVector)
    return loglikelihood(posterior_problem(post, x), post.alg)
end

function LogDensityProblems.logdensity(post::AshDiebackPosterior, x::AbstractVector)
    lp = logprior(post, x)
    isfinite(lp) || return -Inf
    return lp + loglikelihood(post, x)
end

LogDensityProblems.dimension(post::AshDiebackPosterior) = length(post.free) + free_cap_exponent(post)

function LogDensityProblems.capabilities(::Type{<:AshDiebackPosterior})
    return LogDensityProblems.LogDensityOrder{0}()
end
