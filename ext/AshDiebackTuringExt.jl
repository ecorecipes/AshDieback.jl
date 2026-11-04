module AshDiebackTuringExt

using AshDieback: AshDieback, AshDiebackPosterior
using Turing: Turing, @model, Uniform, product_distribution

@model function ashdieback_turing(post, lower, upper)
    θ ~ product_distribution(Uniform.(lower, upper))
    Turing.@addlogprob! AshDieback.loglikelihood(post, θ)
end

function AshDieback.turing_model(post::AshDiebackPosterior)
    return ashdieback_turing(post, post.ranges[post.free, 1], post.ranges[post.free, 2])
end

end
