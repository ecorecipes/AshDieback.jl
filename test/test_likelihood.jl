# The oracle is PEOC 1.0 on Python 2.7. Different summation orders and solvers allow
# rounding-level differences: a relative 1e-10 is the target for another language.
@testset "likelihood" begin
    ref = REFERENCE["loglikelihood"]
    prob = AshDiebackProblem(DATA, ref["parameters"][1])
    for (p, value, zero) in zip(ref["parameters"], ref["value"], ref["zero"])
        sol = solve(remake(prob; p))
        ll, z = peoc_loglikelihood(sol)
        @test z == zero
        @test ll ≈ value rtol = 1e-10
        @test loglikelihood(sol) ≈ value rtol = 1e-10
    end

    @testset "fields, including tails" begin
        f = REFERENCE["fungi"]
        sol = solve(remake(prob; p = ref["parameters"][3]))
        @test size(sol.fungi) == (68, 67, 12)
        @test size(sol.symptoms) == (68, 67, 12)
        @test count(>(0), sol.fungi) == f["positive"]
        @test minimum(filter(>(0), sol.fungi)) ≈ f["min_positive"] rtol = 1e-10
        @test sum(sol.fungi) ≈ f["sum"] rtol = 1e-12
        for (x, y, a, v) in f["probes"]
            @test sol.fungi[Int(x), Int(y), Int(a)] ≈ v rtol = 1e-10
        end
    end

    @testset "model pieces" begin
        θ = AshDiebackParameters(ref["parameters"][3])
        @test θ.D == θ[5] == 25.0
        @test collect(θ) == ref["parameters"][3]
        @test AshDiebackParameters(; (k => θ[i] for (i, k) in enumerate(parameter_names()))...) == θ
        @test temperature_effect(0.0, 30.0, 0.5) == 1.0
        @test temperature_effect(40.0, 30.0, 0.5) == 0.0
        @test temperature_effect(40.0, 30.0, 0.0) == 1.0  # 0^0
        @test temperature_effect(15.0, 30.0, 2.0) ≈ 0.25
        p = remake(prob; p = θ)
        F1, F2 = initial_fungi(p)
        @test F1 ≈ θ.C_init * θ.r_symp .* DATA.psi[1] ./
                   ifelse.(temperature_effect.(DATA.temperature[:, :, 1], θ.gamma, θ.kappa) .== 0,
                           1.0, temperature_effect.(DATA.temperature[:, :, 1], θ.gamma, θ.kappa))
        @test simulated_years(p.settings) == 2007:2018
        @test survey_years(p.settings) == 2008:2019
    end

    @testset "zero likelihood and capping" begin
        # beta_0 = 0: no spores, so no symptoms after the imputed years: q = 0 where
        # trees are infected
        θ = AshDiebackParameters(ref["parameters"][2])
        z = remake(prob; p = AshDiebackParameters([θ[1:2]; 0.0; θ[4:10]]))
        sol = solve(z)
        @test loglikelihood(sol) == -Inf
        @test peoc_loglikelihood(sol)[2]
        # C_init so large that the imputed probability exceeds 1
        big = AshDiebackParameters([θ[1:8]; 150.0; θ[10]])
        @test isnan(peoc_loglikelihood(solve(remake(prob; p = big)))[1])
        @test loglikelihood(remake(prob; p = big)) == -Inf
        capped = AshDiebackProblem(DATA, big; settings = ModelSettings(;
                                                                       cap_symptom_probability = true))
        @test maximum(solve(capped).symptoms) <= 1
    end
end
