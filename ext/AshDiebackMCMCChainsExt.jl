module AshDiebackMCMCChainsExt

using AbstractMCMC: AbstractMCMC
using AshDieback: AshDiebackPosterior, ComponentwiseGammaMH, GammaMHTransition,
                  free_parameters
using LogDensityProblems: LogDensityProblems
using MCMCChains: MCMCChains

parameter_labels(post::AshDiebackPosterior) = free_parameters(post)
parameter_labels(ld) = [Symbol("θ[$i]") for i in 1:LogDensityProblems.dimension(ld)]

function AbstractMCMC.bundle_samples(ts::Vector{GammaMHTransition},
                                     model::AbstractMCMC.LogDensityModel,
                                     ::ComponentwiseGammaMH, state, ::Type{MCMCChains.Chains};
                                     param_names = nothing, discard_initial = 0,
                                     thinning = 1, kwargs...)
    names = param_names === nothing ? parameter_labels(model.logdensity) :
            Symbol.(param_names)
    params = reduce(vcat, (t.params' for t in ts))
    vals = hcat(params, [t.lp for t in ts], Float64[t.accepted for t in ts])
    return MCMCChains.Chains(vals, vcat(names, [:lp, :accepted]),
                             (parameters = names, internals = [:lp, :accepted]);
                             start = discard_initial + 1, thin = thinning)
end

end
