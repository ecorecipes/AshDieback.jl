using LogDensityProblems: LogDensityProblems
using Distributions: MvNormal, logpdf
using LinearAlgebra: Symmetric

# Reference values from the Python 3 port of PEOC, verified identical to PEOC 1.0.
@testset "validation against PEOC" begin
    V = TOML.parsefile(joinpath(@__DIR__, "reference", "peoc_validation.toml"))

    @testset "RMSE and the years after the fit (Table 4)" begin
        r = V["rmse"]
        settings = ModelSettings(; final_year = 2023)
        base = AshDiebackProblem(DATA, r["parameters"])
        prob = remake(base; settings)
        @test prob.settings.final_year == 2023
        @test remake(base; settings = ModelSettings()).laplacian === base.laplacian
        sol = solve(prob)
        @test size(sol.symptoms, 3) == 16
        @test loglikelihood(sol) ≈ r["loglikelihood_2008_2023"] rtol = 1e-10
        t = rmse(sol)
        @test t.years == r["years"]
        @test t.observations == r["observations"]
        @test t.annual ≈ r["annual"] rtol = 1e-10
        @test t.cumulative ≈ r["cumulative"] rtol = 1e-10
        @test sum(t.observations) == 5999
        # the paper's Table 4, as printed. It used the unrounded θmax and data with one more
        # plot in 2020 and 2021, so two values differ in the last digit.
        annual = [0.2222, 0.1465, 0.1898, 0.2387, 0.1940, 0.2621, 0.2739, 0.2720, 0.2437,
                  0.2956, 0.2484, 0.2207, 0.3071, 0.3286, 0.3413, 0.3819]
        cumulative = [0.2222, 0.1641, 0.1735, 0.1954, 0.1951, 0.2062, 0.2170, 0.2249, 0.2267,
                      0.2323, 0.2331, 0.2328, 0.2381, 0.2421, 0.2455, 0.2487]
        @test findall(round.(t.annual; digits = 4) .!= annual) == [13]
        @test findall(round.(t.cumulative; digits = 4) .!= cumulative) == [5]
        @test maximum(abs.(t.annual .- annual)) < 1e-4
        @test maximum(abs.(t.cumulative .- cumulative)) < 1e-4
        @test contains(sprint(show, MIME"text/plain"(), t), "2012   515  0.1940  0.1952")
    end

    @testset "visits behind the means" begin
        for o in DATA.observations
            @test length(o.visit_x) == sum(o.visits)
            @test sum(o.visit_proportion) ≈ sum(o.proportion .* o.visits) rtol = 1e-12
        end
    end

    @testset "likelihood grid (action 8)" begin
        g = V["grid"]
        θ = AshDiebackParameters(; r_symp = 1000.0, r_allee = 0.0, beta_0 = 20.0, beta_1 = 0.0,
                                 D = 16.0, S_sat = 90.0, gamma = 21.50001, C_pers = 1.0,
                                 C_init = 0.0085, kappa = 0.05)
        prob = AshDiebackProblem(DATA, θ)
        res = likelihood_grid(prob, (; beta_0 = [20.0, 26.0], D = [16.0, 20.0],
                                     kappa = [0.05, 0.07]); threaded = false)
        @test size(res.loglikelihood) == (2, 2, 2)
        for (b, d, k, v) in zip(g["beta_0"], g["D"], g["kappa"], g["loglikelihood"])
            c = (findfirst(==(b), res.axes[1]), findfirst(==(d), res.axes[2]),
                 findfirst(==(k), res.axes[3]))
            @test res.loglikelihood[c...] ≈ v rtol = 1e-10
        end
        threaded = likelihood_grid(prob, (; beta_0 = [20.0, 26.0], D = [16.0]); threaded = true)
        @test threaded.loglikelihood[:, 1] == [res.loglikelihood[1, 1, 1], res.loglikelihood[2, 1, 1]]
        # profiles: maximised over the rest, dimensions in the order asked for
        @test profile_likelihood(res, :beta_0, :D, :kappa) == res.loglikelihood
        @test profile_likelihood(res, :D, :beta_0)[2, 1] == maximum(res.loglikelihood[1, 2, :])
        @test profile_likelihood(res, :D)[2] == maximum(res.loglikelihood[:, 2, :])
        best, ll = best_parameters(res)
        @test ll == maximum(res.loglikelihood)
        @test loglikelihood(remake(prob; p = best)) == ll
        @test best.S_sat == 90.0
        @test prod(length, values(peoc_grid_axes())) == 630_000
        @test_throws ArgumentError likelihood_grid(prob, (; nonsense = [1.0]))
        @test_throws ArgumentError likelihood_grid(prob, (; D = Float64[]))
        @test_throws ArgumentError profile_likelihood(res, :gamma)
    end

    @testset "AMIS replayed on PEOC's draws (action 2)" begin
        a = V["amis"]
        n, N = a["iterations"], a["samples"]
        X = reduce(hcat, a["draws"])
        θ = AshDiebackParameters([1000.0, 0.0, 24.0, 0.0, 18.0, 90.0, 21.50001, 1.0, 0.0085, 0.05])
        post = AshDiebackPosterior(AshDiebackProblem(DATA, θ), reduced_model_prior_ranges())
        λ = Float64.(a["log_target"])
        # PEOC's log-likelihood times prior, at a third of its draws
        for i in 1:3:length(λ)
            @test LogDensityProblems.logdensity(post, X[:, i]) ≈ λ[i] rtol = 1e-10
        end
        # threaded evaluation is reproducible
        ks = 1:8
        v = Vector{Float64}(undef, length(ks))
        Threads.@threads for i in ks
            v[i] = LogDensityProblems.logdensity(post, X[:, i])
        end
        @test v == [LogDensityProblems.logdensity(post, X[:, i]) for i in ks]
        # the weights and proposal update, given PEOC's draws and proposals
        means = a["means"]
        covs = [reshape(Float64.(c), 5, 5) for c in a["covariances"]]
        qs = [MvNormal(means[j], Symmetric(covs[j])) for j in 1:n]
        logφ = [[logpdf(q, X[:, i]) for q in qs] for i in axes(X, 2)]
        w = AshDieback.amis_weights(λ, logφ, n)
        @test w ≈ a["weights"] rtol = 1e-8
        m, Σ = AshDieback.weighted_moments(X, Float64.(a["weights"]))
        @test m ≈ means[n + 1] rtol = 1e-12
        @test Σ ≈ covs[n + 1] rtol = 1e-12
    end
end
