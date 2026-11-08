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

    @testset "soft spore cap" begin
        S = 92.0
        ws = [0.0, 1e-120, 1.0, 50.0, 92.0, 200.0, 1e4, 1e300]
        @test saturate.(ws, S) == min.(ws, S)
        @test saturate.(ws, S, Inf) == min.(ws, S)
        @test saturate(S, S, 1.0) ≈ S / 2
        @test saturate(S, S, 4.0) ≈ S / 2^(1 / 4)
        @test saturate(10.0, S, 1.0) ≈ 10.0 * S / (S + 10.0)
        for n in (0.5, 1.0, 4.0, 50.0)
            F = saturate.(ws, S, n)
            @test all(isfinite, F) && issorted(F) && all(F .<= min.(ws, S))
            @test F[2] ≈ 1e-120 rtol = 1e-12   # a load far below the cap is untouched
            @test F[end] ≈ S
        end
        @test saturate(200.0, S, 1e6) ≈ S rtol = 1e-6
        @test saturate(50.0, S, 1e6) ≈ 50.0 rtol = 1e-6
        @test_throws ArgumentError ModelSettings(; saturation_sharpness = 0.0)
        # Sharp enough, the soft cap gives the paper's likelihood; softer, it changes it.
        prob = AshDiebackProblem(DATA, [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.00855, 0.056])
        sharp = remake(prob; settings = ModelSettings(; saturation_sharpness = 1e6))
        soft = remake(prob; settings = ModelSettings(; saturation_sharpness = 2.0))
        ll = loglikelihood(prob)
        @test loglikelihood(sharp) ≈ ll rtol = 1e-6
        @test abs(loglikelihood(soft) - ll) > 1
    end

    @testset "power law in place of the cap" begin
        S = 92.0
        lin = ModelSettings(; spore_exponent = 1.0)
        half = ModelSettings(; spore_exponent = 0.5)
        @test AshDieback.rachis_response(50.0, S, ModelSettings()) == 50.0
        @test AshDieback.rachis_response(500.0, S, ModelSettings()) == S
        @test AshDieback.rachis_response(500.0, S, ModelSettings(; saturation_sharpness = 2.0)) == saturate(500.0, S, 2.0)
        @test AshDieback.rachis_response(500.0, S, lin) ≈ 500.0   # a = 1: no cap at all
        @test AshDieback.rachis_response(S, S, half) ≈ S
        @test AshDieback.rachis_response(4S, S, half) ≈ 2S
        @test AshDieback.rachis_response(1e-120, S, half) ≈ sqrt(1e-120 * S) rtol = 1e-12
        @test AshDieback.rachis_response(0.0, S, half) == 0.0
        @test_throws ArgumentError ModelSettings(; spore_exponent = 0.0)
        @test_throws ArgumentError ModelSettings(; spore_exponent = -1.0)
        @test_throws ArgumentError ModelSettings(; spore_exponent = Inf)
        @test_throws ArgumentError ModelSettings(; spore_exponent = 0.5, saturation_sharpness = 2.0)
        # The field follows the power law of the diffused load: with a = 1, F is the load itself,
        # and it exceeds S where the hard cap would bind.
        θ = [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.00855, 0.056]
        prob = AshDiebackProblem(DATA, θ; settings = ModelSettings(; final_year = 2012))
        hard = solve(prob).fungi
        free = solve(remake(prob; settings = ModelSettings(; final_year = 2012, spore_exponent = 1.0))).fungi
        @test free[:, :, 1:2] == hard[:, :, 1:2]
        @test all(free[:, :, 3] .>= hard[:, :, 3])
        @test maximum(free[:, :, 3]) > S
        below = hard[:, :, 3] .< S
        @test free[:, :, 3][below] == hard[:, :, 3][below]
        @test isfinite(loglikelihood(AshDiebackProblem(DATA, [1000.0, 0.0, 23.26, 0.0, 18.51, 2.0, 21.50001, 1.0, 0.00855, 0.056];
                                                       settings = ModelSettings(; spore_exponent = 0.5))))
    end

    @testset "broken power law above the cap" begin
        S = 92.0
        resp(w, b) = AshDieback.rachis_response(w, S, ModelSettings(; cap_exponent = b))
        ws = [0.0, 1e-120, 1.0, 50.0, 92.0, 200.0, 1e4, 1e300]
        @test resp.(ws, 0.0) == min.(ws, S)
        for b in (0.1, 0.3, 0.5, 1.0)
            F = resp.(ws, b)
            @test F[1:5] == ws[1:5]                  # untouched up to the cap
            @test issorted(F) && all(isfinite, F) && all(F .<= ws)
            @test F[6] > S
        end
        @test resp(4S, 0.5) ≈ 2S
        @test resp(200.0, 1.0) == 200.0
        @test resp(200.0, 0.3) ≈ S * (200.0 / S)^0.3
        @test_throws ArgumentError ModelSettings(; cap_exponent = -0.1)
        @test_throws ArgumentError ModelSettings(; cap_exponent = 1.5)
        @test_throws ArgumentError ModelSettings(; cap_exponent = 0.3, saturation_sharpness = 2.0)
        @test_throws ArgumentError ModelSettings(; cap_exponent = 0.3, spore_exponent = 0.5)
        θ = [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.00855, 0.056]
        prob = AshDiebackProblem(DATA, θ)
        @test loglikelihood(remake(prob; settings = ModelSettings(; cap_exponent = 0.0))) == loglikelihood(prob)
        hard = solve(prob).fungi
        broken = solve(remake(prob; settings = ModelSettings(; cap_exponent = 0.3))).fungi
        # the same imputed years; from the same rachis, the first simulated year differs only above the cap
        @test broken[:, :, 1:2] == hard[:, :, 1:2]
        above = hard[:, :, 3] .== S
        @test any(above)
        @test broken[:, :, 3][.!above] == hard[:, :, 3][.!above]
        @test all(broken[:, :, 3][above] .> S)
        @test sum(broken) > sum(hard)
        @test abs(loglikelihood(remake(prob; settings = ModelSettings(; cap_exponent = 0.3))) - loglikelihood(prob)) > 1
    end

    @testset "recovery with summer heat" begin
        @test persistence(0.95, 30.0, 21.5, 0.0) === 0.95
        @test persistence(0.95, 0.0, 21.5, 0.7) == 0.95
        @test persistence(0.95, -1.0, 21.5, 0.7) == 0.95   # fill value
        @test persistence(1.0, 10.75, 21.5, 0.6) ≈ 0.7
        @test persistence(0.9, 40.0, 21.5, 1.0) == 0.0
        @test_throws ArgumentError ModelSettings(; heat_recovery = 1.5)
        @test_throws ArgumentError ModelSettings(; heat_recovery_lag = 2)
        prob = AshDiebackProblem(DATA, [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.00855, 0.056])
        base = solve(prob).symptoms
        # by hand: q_a = q̃ (1 − C q) + C q with C from the same summer's index (lag 0)
        lag0 = solve(remake(prob; settings = ModelSettings(; heat_recovery = 0.5, heat_recovery_lag = 0)))
        sol = solve(prob)
        t0 = 2007 - first(DATA.climate_years) + 1
        i, j = findmax(DATA.temperature[:, :, t0 + 3] .* (base[:, :, 2] .> 0.01))[2].I
        @test DATA.temperature[i, j, t0 + 3] > 0
        q = 0.0
        for l in 1:3
            qt = sol.fungi[i, j, l] * temperature_effect(DATA.temperature[i, j, t0 + l - 1], 21.50001, 0.056) / 1000
            Ce = 1 - 0.5 * clamp(DATA.temperature[i, j, t0 + l] / 21.50001, 0, 1)
            q = qt * (1 - Ce * q) + Ce * q
        end
        @test lag0.symptoms[i, j, 3] ≈ q rtol = 1e-12
        @test all(lag0.symptoms .<= base .+ 1e-15)
        @test lag0.symptoms[i, j, 3] < base[i, j, 3]
    end
end
