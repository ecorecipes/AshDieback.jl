using LogDensityProblems: LogDensityProblems

@testset "posterior" begin
    θ = REFERENCE["loglikelihood"]["parameters"][2]
    prob = AshDiebackProblem(DATA, θ)
    post = AshDiebackPosterior(prob, reduced_model_prior_ranges())
    @test free_parameters(post) == [:beta_0, :D, :S_sat, :C_init, :kappa]
    @test LogDensityProblems.dimension(post) == 5
    x = θ[post.free]
    @test full_parameters(post, x) == AshDiebackParameters([1000, 0, 30, 0, 20, 90,
                                                            21.50001, 1, 0.0085, 0.1])
    @test logprior(post, x) ≈ -log(800 * 200 * 500 * 200 * 10)
    @test logprior(post, [x[1:4]; -1.0]) == -Inf
    @test LogDensityProblems.logdensity(post, [x[1:4]; 11.0]) == -Inf
    ld = LogDensityProblems.logdensity(post, x)
    @test ld ≈ logprior(post, x) + loglikelihood(post, x)
    @test LogDensityProblems.capabilities(post) == LogDensityProblems.LogDensityOrder{0}()
    full = AshDiebackPosterior(prob)
    @test length(free_parameters(full)) == 9
    @test peoc_prior_ranges()[1, :] == [1000.0, 1000.0]

    @testset "free cap exponent" begin
        bpost = AshDiebackPosterior(prob, reduced_model_prior_ranges(); cap_exponent = (0.0, 1.0))
        @test free_parameters(bpost) == [:beta_0, :D, :S_sat, :C_init, :kappa, :cap_exponent]
        @test LogDensityProblems.dimension(bpost) == 6
        @test parameter_bounds(bpost) == ([0.0, 0, 0, 0, 0, 0], [800.0, 200, 500, 200, 10, 1])
        @test parameter_bounds(post) == ([0.0, 0, 0, 0, 0], [800.0, 200, 500, 200, 10])
        xb = [x; 0.3]
        @test full_parameters(bpost, xb) == full_parameters(post, x)
        @test logprior(bpost, xb) ≈ logprior(post, x)
        @test logprior(bpost, [x; 1.5]) == -Inf
        @test LogDensityProblems.logdensity(bpost, [x; -0.1]) == -Inf
        bprob = posterior_problem(bpost, xb)
        @test bprob.settings.cap_exponent == 0.3
        @test bprob.laplacian === prob.laplacian && bprob.binomial === prob.binomial
        fresh = AshDiebackProblem(DATA, full_parameters(post, x); settings = ModelSettings(; cap_exponent = 0.3))
        @test loglikelihood(bpost, xb) ≈ loglikelihood(fresh) rtol = 1e-12
        @test loglikelihood(bpost, xb) != loglikelihood(post, x)
        @test loglikelihood(bpost, [x; 0.0]) ≈ loglikelihood(post, x) rtol = 1e-12
        @test_throws ArgumentError full_parameters(bpost, x)
        @test_throws ArgumentError logprior(post, xb)
        @test_throws ArgumentError AshDiebackPosterior(prob; cap_exponent = (0.5, 0.2))
        @test_throws ArgumentError AshDiebackPosterior(prob; cap_exponent = (0.0, 2.0))
        soft = remake(prob; settings = ModelSettings(; saturation_sharpness = 2.0))
        @test soft.binomial === prob.binomial
        @test_throws ArgumentError AshDiebackPosterior(soft; cap_exponent = (0.0, 1.0))
        @test length(remake(prob; settings = ModelSettings(; final_year = 2018)).binomial) == length(prob.binomial) - 1
        @test ComponentwiseGammaMH(bpost).shape == fill(200.0, 6)
        @test ComponentwiseGammaMH(bpost; shape = [fill(200.0, 10); 50.0]).shape[end] == 50.0
    end
end
