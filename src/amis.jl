"""
    DegenerateProposalError(iteration, message)

The Gaussian proposal of [`amis`](@ref) became degenerate at `iteration`: the weights
collapsed onto fewer samples than there are free parameters, so the weighted covariance is
singular. Use more samples per iteration, or fix more parameters.
"""
struct DegenerateProposalError <: AshDiebackError
    iteration::Int
    message::String
end

function Base.showerror(io::IO, e::DegenerateProposalError)
    return print(io, "DegenerateProposalError at AMIS iteration ", e.iteration, ": ",
                 e.message)
end

"""
    AMISResult

Output of [`amis`](@ref).

Fields:
- `samples::Vector{Matrix{Float64}}`: the draws of each iteration, one column per draw.
- `logdensity::Vector{Vector{Float64}}`: the target log density at each draw (`-Inf` for
  draws outside the prior or with zero likelihood).
- `weights::Vector{Vector{Float64}}`: the final normalised importance weights of every
  draw, with the same layout as `logdensity`.
- `means::Vector{Vector{Float64}}`, `covariances::Vector{Matrix{Float64}}`: the proposal of
  each iteration, and the one the last iteration produced.
"""
struct AMISResult
    samples::Vector{Matrix{Float64}}
    logdensity::Vector{Vector{Float64}}
    weights::Vector{Vector{Float64}}
    means::Vector{Vector{Float64}}
    covariances::Vector{Matrix{Float64}}
end

"""
    amis([rng], target, mean, covariance; iterations, samples,
         threaded = Threads.nthreads() > 1, callback = nothing)
    amis([rng], target, previous::AMISResult; iterations, samples, ...)

Adaptive multiple importance sampling (Cornuet et al. 2012), as used by PEOC for the paper's
posterior. `target` is a LogDensityProblems.jl log density, such as an
[`AshDiebackPosterior`](@ref); `mean` and `covariance` define the initial Gaussian proposal
(covariance entries are variances, as PEOC's `scale_para_AMIS` on the diagonal).

Each iteration draws `samples` points from the current proposal N(mₖ, Σₖ) and evaluates the
target there (in parallel threads if `threaded`). It then recomputes the deterministic-mixture
weights of all draws so far, wᵢ ∝ π(θᵢ) / ((1/(k+1)) Σⱼ φ(θᵢ; mⱼ, Σⱼ)), and moves the proposal
to their weighted mean and covariance. The mixture is evaluated on the log scale, so it is
better conditioned than PEOC's; the algorithm is the same. Throws a
[`DegenerateProposalError`](@ref) if the proposal covariance becomes singular.

With `previous`, it continues that run for `iterations` more iterations, which given the same
`rng` state matches a single longer run. `callback(result, k)` is called after iteration `k`,
for example to save a checkpoint; `result` is updated in place, so copy what must be kept.
"""
function amis(rng::AbstractRNG, target, mean::AbstractVector, covariance::AbstractMatrix;
              kwargs...)
    d = LogDensityProblems.dimension(target)
    length(mean) == d && size(covariance) == (d, d) ||
        throw(ArgumentError("mean and covariance must have dimension $d"))
    res = AMISResult(Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[],
                     [Vector{Float64}(mean)], [Matrix{Float64}(covariance)])
    return amis(rng, target, res; kwargs...)
end

function amis(rng::AbstractRNG, target, previous::AMISResult; iterations::Integer,
              samples::Integer, threaded::Bool = Threads.nthreads() > 1, callback = nothing)
    iterations >= 1 && samples >= 1 ||
        throw(ArgumentError("iterations and samples must be positive"))
    res = AMISResult(copy(previous.samples), copy(previous.logdensity),
                     copy(previous.weights), copy(previous.means), copy(previous.covariances))
    done = length(res.samples)
    # log densities of every earlier draw under every earlier proposal
    proposals = [amis_proposal(res.means[j], res.covariances[j], j) for j in 1:done]
    logφ = Vector{Float64}[]
    for X in res.samples, i in axes(X, 2)
        x = view(X, :, i)
        push!(logφ, [logpdf(q, x) for q in proposals])
    end
    for k in (done + 1):(done + iterations)
        q = amis_proposal(res.means[end], res.covariances[end], k)
        push!(proposals, q)
        X = rand(rng, q, samples)
        λ = Vector{Float64}(undef, samples)
        if threaded
            Threads.@threads :dynamic for i in 1:samples
                λ[i] = LogDensityProblems.logdensity(target, view(X, :, i))
            end
        else
            for i in 1:samples
                λ[i] = LogDensityProblems.logdensity(target, view(X, :, i))
            end
        end
        push!(res.samples, X)
        push!(res.logdensity, λ)
        Xall = reduce(hcat, res.samples)
        for (i, v) in enumerate(logφ)  # old draws under the new proposal
            push!(v, logpdf(q, view(Xall, :, i)))
        end
        for i in (size(Xall, 2) - samples + 1):size(Xall, 2)  # new draws under all proposals
            x = view(Xall, :, i)
            push!(logφ, [logpdf(p, x) for p in proposals])
        end
        w = amis_weights(reduce(vcat, res.logdensity), logφ, k)
        m, Σ = weighted_moments(Xall, w)
        push!(res.means, m)
        push!(res.covariances, Σ)
        empty!(res.weights)
        offset = 0
        for λk in res.logdensity
            push!(res.weights, w[(offset + 1):(offset + length(λk))])
            offset += length(λk)
        end
        callback === nothing || callback(res, k)
    end
    return res
end

function amis(target, mean_or_previous, args...; kwargs...)
    return amis(Random.default_rng(), target, mean_or_previous, args...; kwargs...)
end

function amis_proposal(m, Σ, k)
    try
        return MvNormal(m, Symmetric(Σ))
    catch e
        e isa PosDefException || rethrow()
        throw(DegenerateProposalError(k, "the proposal covariance is not positive definite"))
    end
end

# The deterministic-mixture weights of PEOC's compute_weight, given each draw's target log
# density and its log density under each of the k proposals.
function amis_weights(λ::AbstractVector, logφ::AbstractVector, k::Integer)
    logw = [isfinite(λ[i]) ? λ[i] - (logsumexp(logφ[i]) - log(k)) : -Inf for i in eachindex(λ)]
    any(isfinite, logw) ||
        throw(DegenerateProposalError(k, "every draw has zero target density"))
    w = exp.(logw .- maximum(logw))
    return w ./ sum(w)
end

# PEOC's compute_mean_variance: the plain weighted mean and covariance of the columns of X.
function weighted_moments(X::AbstractMatrix, w::AbstractVector)
    m = X * w
    Xc = X .- m
    Σ = (Xc .* w') * Xc'
    return m, (Σ + Σ') / 2
end

"""
    amis_draws(res::AMISResult)

All the draws of `res` as `(samples, logdensity, weights)`: a matrix with one column per
draw, and two vectors.
"""
function amis_draws(res::AMISResult)
    return reduce(hcat, res.samples), reduce(vcat, res.logdensity), reduce(vcat, res.weights)
end

"""
    effective_sample_size(res::AMISResult)

Kish's effective sample size of the final weights, 1 / Σ wᵢ².
"""
effective_sample_size(res::AMISResult) = 1 / sum(sum(abs2, w) for w in res.weights)

"""
    best_sample(res::AMISResult)

The draw of `res` with the highest target log density, as `(x, logdensity)`. The paper's
θmax (eq. 14) is the best sample of its AMIS run.
"""
function best_sample(res::AMISResult)
    X, λ, _ = amis_draws(res)
    i = argmax(λ)
    return X[:, i], λ[i]
end

"""
    weighted_quantile(x, w, p)

The `p` quantile of the values `x` with weights `w`: the smallest value whose cumulative
normalised weight reaches `p`. A helper for marginal summaries of an [`AMISResult`](@ref).
"""
function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1], not $p"))
    length(x) == length(w) || throw(ArgumentError("x and w must have the same length"))
    order = sortperm(x)
    total = sum(w)
    c = 0.0
    for i in order
        c += w[i]
        c >= p * total && return x[i]
    end
    return x[order[end]]
end
