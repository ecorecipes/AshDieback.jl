"""
    peoc_mh_shapes()

The gamma proposal shapes of PEOC 1.0's Metropolis–Hastings (`scale_para_MH`), one per
parameter in the order of [`AshDiebackParameters`](@ref): 100 for `r_symp` and 200 for the
others. A proposal from θⱼ has mean θⱼ and standard deviation θⱼ/√shape.
"""
peoc_mh_shapes() = [100.0; fill(200.0, 9)]

"""
    ComponentwiseGammaMH(shape; blocked = false)
    ComponentwiseGammaMH(post::AshDiebackPosterior; shape = peoc_mh_shapes(), blocked = false)

The Metropolis–Hastings sampler of PEOC, as an `AbstractMCMC.AbstractSampler` for any
LogDensityProblems.jl log density over positive parameters (wrap it in
`AbstractMCMC.LogDensityModel`, or pass an [`AshDiebackPosterior`](@ref) to `sample`).

Each iteration proposes θ′ⱼ ~ Gamma(shape = sⱼ, scale = θⱼ/sⱼ), a multiplicative random walk
with mean θⱼ, for one parameter, cycling through the parameters in order (or for all of them
at once with `blocked = true`), and accepts it with the Metropolis–Hastings probability,
which corrects for the asymmetric proposal. A parameter at exactly 0 cannot move.

`shape` gives sⱼ for each free parameter; the second form takes the shapes of the free
parameters of `post` from a 10-vector, with an 11th entry (default 200) for a free cap
exponent. `sample` needs `initial_params`.
"""
struct ComponentwiseGammaMH <: AbstractMCMC.AbstractSampler
    shape::Vector{Float64}
    blocked::Bool
end

function ComponentwiseGammaMH(shape::AbstractVector{<:Real}; blocked::Bool = false)
    all(>(0), shape) || throw(ArgumentError("gamma shapes must be positive"))
    return ComponentwiseGammaMH(Vector{Float64}(shape), blocked)
end

function ComponentwiseGammaMH(post::AshDiebackPosterior; shape = peoc_mh_shapes(),
                              blocked::Bool = false)
    n = 10 + free_cap_exponent(post)
    length(shape) in (10, n) ||
        throw(ArgumentError("shape must have one entry per parameter ($n)"))
    s = shape[post.free]
    free_cap_exponent(post) && push!(s, length(shape) == 11 ? shape[11] : 200.0)
    return ComponentwiseGammaMH(s; blocked)
end

"""
    GammaMHTransition

One draw of [`ComponentwiseGammaMH`](@ref): the parameters `params`, the log density `lp`
there, and whether the proposal of this iteration was `accepted`.
"""
struct GammaMHTransition
    params::Vector{Float64}
    lp::Float64
    accepted::Bool
end

"""
    GammaMHState

State of [`ComponentwiseGammaMH`](@ref): the current `params` and log density `lp`, the
`component` to update next, and the number of accepted proposals out of `iteration`.
"""
struct GammaMHState
    params::Vector{Float64}
    lp::Float64
    component::Int
    accepted::Int
    iteration::Int
end

AbstractMCMC.getparams(state::GammaMHState) = state.params

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel,
                           spl::ComponentwiseGammaMH; initial_params = nothing, kwargs...)
    initial_params === nothing &&
        throw(ArgumentError("ComponentwiseGammaMH needs initial_params"))
    d = LogDensityProblems.dimension(model.logdensity)
    length(initial_params) == d ||
        throw(ArgumentError("initial_params has $(length(initial_params)) entries; the " *
                            "log density has dimension $d"))
    length(spl.shape) == d ||
        throw(ArgumentError("the sampler has $(length(spl.shape)) shapes for dimension $d"))
    x = Vector{Float64}(initial_params)
    lp = LogDensityProblems.logdensity(model.logdensity, x)
    return GammaMHTransition(x, lp, true), GammaMHState(x, lp, 1, 0, 0)
end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel,
                           spl::ComponentwiseGammaMH, state::GammaMHState; kwargs...)
    x = state.params
    y = copy(x)
    components = spl.blocked ? eachindex(x) : (state.component:state.component)
    logq = 0.0  # log q(x | y) - log q(y | x)
    for j in components
        x[j] > 0 || continue
        s = spl.shape[j]
        y[j] = rand(rng, Gamma(s, x[j] / s))
        logq += y[j] > 0 ?
                logpdf(Gamma(s, y[j] / s), x[j]) - logpdf(Gamma(s, x[j] / s), y[j]) : -Inf
    end
    lpy = isfinite(logq) ? LogDensityProblems.logdensity(model.logdensity, y) : -Inf
    # a state with zero density accepts any proposal with positive density
    accept = log(rand(rng)) < lpy - state.lp + logq
    next = spl.blocked ? 1 : mod1(state.component + 1, length(x))
    if accept
        new = GammaMHState(y, lpy, next, state.accepted + 1, state.iteration + 1)
    else
        new = GammaMHState(x, state.lp, next, state.accepted, state.iteration + 1)
    end
    return GammaMHTransition(new.params, new.lp, accept), new
end

function AbstractMCMC.sample(rng::AbstractRNG, post::AshDiebackPosterior,
                             spl::ComponentwiseGammaMH, N::Integer; kwargs...)
    return AbstractMCMC.sample(rng, AbstractMCMC.LogDensityModel(post), spl, N; kwargs...)
end

function AbstractMCMC.sample(post::AshDiebackPosterior, spl::ComponentwiseGammaMH,
                             N::Integer; kwargs...)
    return AbstractMCMC.sample(Random.default_rng(), post, spl, N; kwargs...)
end
