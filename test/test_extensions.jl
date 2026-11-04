using MCMCChains: MCMCChains
using Turing: Turing
using AbstractMCMC: AbstractMCMC
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

@testset "extensions" begin
    θ = REFERENCE["loglikelihood"]["parameters"][2]
    post = AshDiebackPosterior(AshDiebackProblem(DATA, θ), reduced_model_prior_ranges())
    x0 = θ[post.free]

    @testset "MCMCChains" begin
        chn = AbstractMCMC.sample(Xoshiro(2), post, ComponentwiseGammaMH(post), 4;
                                  initial_params = x0, chain_type = MCMCChains.Chains,
                                  progress = false)
        @test chn isa MCMCChains.Chains
        @test names(chn, :parameters) == free_parameters(post)
        @test size(chn, 1) == 4
        @test chn[:lp][1] ≈ LogDensityProblems.logdensity(post, x0)
    end

    @testset "Turing" begin
        model = turing_model(post)
        # the model's log joint is the posterior's log density
        @test Turing.DynamicPPL.logjoint(model, (; θ = x0)) ≈
              LogDensityProblems.logdensity(post, x0)
    end
end
