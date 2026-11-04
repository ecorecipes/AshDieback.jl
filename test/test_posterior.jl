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
end
