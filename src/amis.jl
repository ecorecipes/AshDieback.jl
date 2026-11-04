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
    amis([rng], target, mean, covariance; iterations, samples, threaded = Threads.nthreads() > 1)

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
"""
function amis(rng::AbstractRNG, target, mean::AbstractVector, covariance::AbstractMatrix;
              iterations::Integer, samples::Integer,
              threaded::Bool = Threads.nthreads() > 1)
    d = LogDensityProblems.dimension(target)
    length(mean) == d && size(covariance) == (d, d) ||
        throw(ArgumentError("mean and covariance must have dimension $d"))
    iterations >= 1 && samples >= 1 ||
        throw(ArgumentError("iterations and samples must be positive"))
    proposals = MvNormal[]
    m, Σ = Vector{Float64}(mean), Matrix{Float64}(covariance)
    res = AMISResult(Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[], [copy(m)],
                     [copy(Σ)])
    logφ = Vector{Float64}[]  # logφ[i][j]: log density of draw i under proposal j
    for k in 1:iterations
        q = try
            MvNormal(m, Symmetric(Σ))
        catch e
            e isa PosDefException || rethrow()
            throw(DegenerateProposalError(k, "the proposal covariance is not positive definite"))
        end
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
        for v in logφ  # old draws under the new proposal
            push!(v, NaN)
        end
        Xall = reduce(hcat, res.samples)
        for (i, v) in enumerate(logφ)
            v[end] = logpdf(q, view(Xall, :, i))
        end
        for i in (size(Xall, 2) - samples + 1):size(Xall, 2)  # new draws under all proposals
            push!(logφ, [logpdf(p, view(Xall, :, i)) for p in proposals])
        end
        λall = reduce(vcat, res.logdensity)
        logw = [isfinite(λall[i]) ? λall[i] - (logsumexp(logφ[i]) - log(k)) : -Inf
                for i in eachindex(λall)]
        any(isfinite, logw) ||
            throw(DegenerateProposalError(k, "every draw has zero target density"))
        w = exp.(logw .- maximum(logw))
        w ./= sum(w)
        m = Xall * w
        Xc = Xall .- m
        Σ = (Xc .* w') * Xc'
        Σ = (Σ + Σ') / 2
        push!(res.means, copy(m))
        push!(res.covariances, copy(Σ))
        empty!(res.weights)
        offset = 0
        for λk in res.logdensity
            push!(res.weights, w[(offset + 1):(offset + length(λk))])
            offset += length(λk)
        end
    end
    return res
end

amis(target, mean, covariance; kwargs...) =
    amis(Random.default_rng(), target, mean, covariance; kwargs...)
