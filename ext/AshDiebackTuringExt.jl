module AshDiebackTuringExt

using AshDieback: AshDieback, AshDiebackPosterior
using Turing: Turing, @model, Uniform, product_distribution

@model function ashdieback_turing(post, lower, upper)
    θ ~ product_distribution(Uniform.(lower, upper))
    Turing.@addlogprob! AshDieback.loglikelihood(post, θ)
end

function AshDieback.turing_model(post::AshDiebackPosterior)
    return ashdieback_turing(post, AshDieback.parameter_bounds(post)...)
end

end
