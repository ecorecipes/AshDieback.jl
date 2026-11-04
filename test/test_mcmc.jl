using AbstractMCMC: AbstractMCMC
using Distributions: LogNormal, Gamma, logpdf
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro
using Statistics: mean, var

# A cheap positive target with known moments, for checking that the sampler is correct.
struct ProductTarget{D}
    dists::D
end
LogDensityProblems.logdensity(t::ProductTarget, x) = sum(logpdf.(t.dists, x))
LogDensityProblems.dimension(t::ProductTarget) = length(t.dists)
LogDensityProblems.capabilities(::Type{<:ProductTarget}) = LogDensityProblems.LogDensityOrder{0}()

@testset "ComponentwiseGammaMH" begin
    target = ProductTarget([LogNormal(0.0, 0.5), Gamma(4.0, 2.0)])
    model = AbstractMCMC.LogDensityModel(target)
    for blocked in (false, true)
        spl = ComponentwiseGammaMH([20.0, 20.0]; blocked)
        chain = AbstractMCMC.sample(Xoshiro(1), model, spl, 60_000; initial_params = [1.0, 5.0],
                                    progress = false)
        X = reduce(hcat, (t.params for t in chain))[:, 10_001:end]
        @test mean(X[1, :]) ≈ mean(LogNormal(0.0, 0.5)) rtol = 0.05
        @test mean(X[2, :]) ≈ 8.0 rtol = 0.05
        @test var(X[2, :]) ≈ 16.0 rtol = 0.15
        rate = mean(t.accepted for t in chain)
        @test 0.2 < rate < 0.95
    end
    # determinism with a seeded RNG
    spl = ComponentwiseGammaMH([20.0, 20.0])
    c1 = AbstractMCMC.sample(Xoshiro(7), model, spl, 50; initial_params = [1.0, 5.0],
                             progress = false)
    c2 = AbstractMCMC.sample(Xoshiro(7), model, spl, 50; initial_params = [1.0, 5.0],
                             progress = false)
    @test [t.params for t in c1] == [t.params for t in c2]
    @test_throws ArgumentError AbstractMCMC.sample(Xoshiro(1), model, spl, 2; progress = false)
    @test_throws ArgumentError ComponentwiseGammaMH([1.0, 0.0])

    @testset "on the ash dieback posterior" begin
        θ = REFERENCE["loglikelihood"]["parameters"][2]
        post = AshDiebackPosterior(AshDiebackProblem(DATA, θ), reduced_model_prior_ranges())
        spl = ComponentwiseGammaMH(post)
        @test spl.shape == fill(200.0, 5)
        chain = AbstractMCMC.sample(Xoshiro(1), post, spl, 6; initial_params = θ[post.free],
                                    progress = false)
        @test length(chain) == 6
        @test chain[1].lp ≈ LogDensityProblems.logdensity(post, θ[post.free])
        # component-wise: each iteration changes at most one parameter
        for k in 2:6
            @test count(chain[k].params .!= chain[k - 1].params) <= 1
        end
    end
end
