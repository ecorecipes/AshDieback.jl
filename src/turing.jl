"""
    turing_model(post::AshDiebackPosterior)

The posterior `post` as a Turing.jl model, with a uniform prior on each free parameter (in
one vector `θ`, ordered as [`free_parameters`](@ref)) and the model's log-likelihood added
with `@addlogprob!`. Requires `using Turing`.

The likelihood has no useful gradient (it is a 4556-cell PDE solve per year, with kinks), so
use a gradient-free sampler, e.g. `sample(turing_model(post), MH(), 1000)`, or pass a
[`ComponentwiseGammaMH`](@ref) through `externalsampler`.
"""
function turing_model end
