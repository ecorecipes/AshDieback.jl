"""
    AshDieback

The mechanistic-statistical model of the spread of ash dieback (*Hymenoscyphus fraxineus*)
across France of Fritsch, Grosdidier, Gégout-Petit and Marçais, with its data and its
Bayesian inference: a Julia port of PEOC (Parameters Estimation Of Chalara model).

A yearly cycle of spore production from colonised leaf debris, spore diffusion and
saturation (the rachis colonisation model) drives a Bernoulli symptom model for each tree,
observed through a binomial survey model. See [`AshDiebackProblem`](@ref) to simulate it,
[`loglikelihood`](@ref) to score it, [`AshDiebackPosterior`](@ref) for the posterior as a
LogDensityProblems.jl density, and [`ComponentwiseGammaMH`](@ref) and [`amis`](@ref) for
PEOC's samplers.
"""
module AshDieback

using AbstractMCMC: AbstractMCMC
using CommonSolve: CommonSolve, solve
using Distributions: Distributions, Gamma, MvNormal, logpdf
using LinearAlgebra: I, PosDefException, Symmetric, mul!
using LinearSolve: CHOLMODFactorization, LinearProblem, init, solve!
using LogDensityProblems: LogDensityProblems
using LogExpFunctions: logsumexp
using Random: Random, AbstractRNG
using RecipesBase: RecipesBase, @recipe
using SciMLBase: SciMLBase, remake
using SparseArrays: SparseMatrixCSC, sparse, spdiagm
using SpecialFunctions: loggamma
import StatsAPI: loglikelihood

include("errors.jl")
export AshDiebackError, DataFormatError, DegenerateProposalError

include("grid.jl")
export QuadratGrid, ncells, quadrat_index

include("data.jl")
export PEOCData, YearObservations, datadir, load_peoc_data

include("parameters.jl")
export AshDiebackParameters, parameter_names, temperature_effect

include("diffusion.jl")
export AbstractDiffusionAlgorithm, SparseCrankNicolson, DenseCrankNicolson,
       neumann_laplacian

include("problem.jl")
export ModelSettings, saturate, persistence, AshDiebackProblem, AshDiebackSolution, BinomialTerms,
       simulated_years, survey_years, initial_fungi, symptom_probability, solve, remake

include("likelihood.jl")
export loglikelihood, peoc_loglikelihood

include("posterior.jl")
export AshDiebackPosterior, peoc_prior_ranges, reduced_model_prior_ranges,
       free_parameters, full_parameters, logprior

include("mcmc.jl")
export ComponentwiseGammaMH, GammaMHTransition, GammaMHState, peoc_mh_shapes

include("amis.jl")
export AMISResult, amis, amis_draws, effective_sample_size, best_sample, weighted_quantile

include("validation.jl")
export RMSETable, rmse, LikelihoodGrid, likelihood_grid, peoc_grid_axes, profile_likelihood,
       best_parameters

include("host.jl")
export HostClass, HostDynamics, HostProblem, HostSolution, constant_host, host_years, live_host,
       symptom_share, AbstractIntervention, Vaccinate, PlantStock, Fell, ReduceLitter,
       vaccinated_class

include("gb.jl")
export GBInputs, load_gb_inputs, load_gb_future, gb_problem

include("plotting.jl")

include("turing.jl")
export turing_model

end
