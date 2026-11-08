using AshDieback: temperature_effect

const THETA_HOST = [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0, 0.00855, 0.056]

# A uniform grid: diffusion leaves a uniform field alone, so one quadrat can be followed by hand.
function uniform_problem(; d = 0.5, psi = 1.0, T = 5.0, θ = THETA_HOST, last_year = 2030, n = Inf, a = nothing, b = 0.0)
    grid = QuadratGrid(; quadrat_length = 10.0, nx_mainland = 4, ny_mainland = 4,
                       border = (0, 0, 0, 0))
    nx, ny = size(grid)
    years = 2000:last_year
    data = PEOCData(grid, years, zeros(nx, ny, length(years)), fill(float(T), nx, ny, length(years)),
                    :T28, fill(float(d), nx, ny), (fill(float(psi), nx, ny), fill(2.0 * psi, nx, ny)),
                    [YearObservations(a, Int[], Int[], Float64[], Int[], Int[], Int[], Float64[])
                     for a in 2000:last_year])
    settings = ModelSettings(; initialization_year = 2001, final_year = last_year, delta_t = 1.0,
                             saturation_sharpness = n, spore_exponent = a, cap_exponent = b)
    return AshDiebackProblem(data, θ; settings)
end

@testset "host dynamics" begin
    @testset "the constant host reproduces the paper's model" begin
        for (C, h, lag) in ((1.0, 0.0, 1), (0.9, 0.0, 1), (1.0, 0.6, 1), (0.9, 0.8, 0))
            θ = copy(THETA_HOST)
            θ[8] = C
            settings = ModelSettings(; delta_t = 1.0, final_year = 2015, heat_recovery = h, heat_recovery_lag = lag)
            prob = AshDiebackProblem(DATA, θ; settings)
            ref = solve(prob)
            sol = solve(HostProblem(prob, constant_host()))
            @test maximum(abs.(sol.fungi .- ref.fungi) ./ max.(abs.(ref.fungi), 1e-300)) < 1e-11
            ash = DATA.ash_density .> 0
            q = symptom_share(sol)
            for l in axes(ref.symptoms, 3)
                @test maximum(abs.(q[:, :, l + 1] .- ref.symptoms[:, :, l])[ash]) < 1e-12
            end
            @test live_host(sol)[:, :, end] ≈ DATA.ash_density
            @test all(iszero, sol.disease_deaths)
        end
    end

    @testset "identical classes make no difference" begin
        prob = AshDiebackProblem(DATA, THETA_HOST; settings = ModelSettings(; delta_t = 1.0, final_year = 2012))
        one = solve(HostProblem(prob, HostDynamics([HostClass(:a; mortality = 0.1)])))
        two = solve(HostProblem(prob, HostDynamics([HostClass(:a; share = 0.3, mortality = 0.1),
                                                    HostClass(:b; share = 0.7, mortality = 0.1)])))
        @test two.fungi ≈ one.fungi rtol = 1e-12
        @test live_host(two) ≈ live_host(one) rtol = 1e-12
    end

    @testset "follows a hand calculation in one quadrat" begin
        μ, mb, g, ρ, J = 0.2, 0.01, 1 / 5, 0.1, 3.0
        host = HostDynamics([HostClass(:ash; mortality = μ)]; background_mortality = mb,
                            juvenile_years = 5, juvenile_mortality_factor = J,
                            juvenile_inoculum = 0.1, regeneration_rate = ρ)
        prob = uniform_problem()
        sol = solve(HostProblem(prob, host))
        θ = prob.p
        K = 0.5
        j0 = mb / (1 - (1 - mb) * (1 - g))
        H, Y, JH, JY = K * (1 - j0), 0.0, K * j0, 0.0
        A0 = K * (1 - j0)
        ft = temperature_effect(5.0, θ.gamma, θ.kappa)
        for l in 1:(length(host_years(sol)) - 1)
            p = sol.fungi[2, 2, l] * ft / θ.r_symp
            A, N = H + Y, H + Y + JH + JY
            ya, ha, yj, hj = Y + p * H, (1 - p) * H, JY + p * JH, (1 - p) * JH
            ya *= (1 - μ) * (1 - mb)
            ha *= 1 - mb
            yj *= (1 - min(1, μ * J)) * (1 - mb)
            hj *= 1 - mb
            H, Y, JH, JY = ha + g * hj, ya + g * yj, (1 - g) * hj, (1 - g) * yj
            JH += A / A0 * mb * N + ρ * A * max(0, 1 - N / K)
            @test sol.healthy[2, 2, 1, l + 1] ≈ H rtol = 1e-12
            @test sol.symptomatic[2, 2, 1, l + 1] ≈ Y rtol = 1e-12
            @test sol.juvenile_healthy[2, 2, 1, l + 1] ≈ JH rtol = 1e-12
            @test sol.juvenile_symptomatic[2, 2, 1, l + 1] ≈ JY rtol = 1e-12
        end
    end

    @testset "the spore source follows the live host" begin
        # Uniform, so diffusion only applies the time-stepping to the source: F(3) = min(β (χ1 + χ2), S)
        prob = uniform_problem(; psi = 0.01)
        host = HostDynamics([HostClass(:ash; mortality = 0.5)])
        sol = solve(HostProblem(prob, host))
        A(l) = sol.healthy[2, 2, 1, l] + sol.symptomatic[2, 2, 1, l]
        for l in 3:8
            expected = prob.p.beta_0 * (sol.fungi[2, 2, l - 2] * A(l - 2) + sol.fungi[2, 2, l - 1] * A(l - 1))
            @test sol.fungi[2, 2, l] ≈ min(expected, prob.p.S_sat) rtol = 1e-10
        end
    end

    @testset "the spore load before a soft cap" begin
        for n in (Inf, 1.0, 4.0)
            prob = uniform_problem(; psi = 0.01, n)
            sol = solve(HostProblem(prob, HostDynamics([HostClass(:ash; mortality = 0.5)])))
            A(l) = sol.healthy[2, 2, 1, l] + sol.symptomatic[2, 2, 1, l]
            @test all(isnan, sol.spore_load[:, :, 1:2])
            for l in 3:8
                expected = prob.p.beta_0 * (sol.fungi[2, 2, l - 2] * A(l - 2) + sol.fungi[2, 2, l - 1] * A(l - 1))
                @test sol.spore_load[2, 2, l] ≈ expected rtol = 1e-10
                @test sol.fungi[2, 2, l] == saturate(sol.spore_load[2, 2, l], prob.p.S_sat, n)
            end
        end
        for a in (0.5, 1.0)
            prob = uniform_problem(; psi = 0.01, a)
            sol = solve(HostProblem(prob, HostDynamics([HostClass(:ash; mortality = 0.5)])))
            for l in 3:8
                @test sol.fungi[2, 2, l] ≈ prob.p.S_sat * (sol.spore_load[2, 2, l] / prob.p.S_sat)^a rtol = 1e-14
            end
        end
        for b in (0.3, 1.0)
            prob = uniform_problem(; psi = 1.0, b)
            sol = solve(HostProblem(prob, HostDynamics([HostClass(:ash; mortality = 0.5)])))
            S, w = prob.p.S_sat, sol.spore_load[2, 2, 3:8]
            @test any(w .> S)   # the load passes the cap, so the broken law is exercised
            @test sol.fungi[2, 2, 3:8] ≈ [x <= S ? x : S * (x / S)^b for x in w] rtol = 1e-14
        end
    end

    @testset "an undiseased stand stays as it started" begin
        prob = uniform_problem(; psi = 0.0)
        host = HostDynamics([HostClass(:a; share = 0.9, mortality = 0.3),
                             HostClass(:b; share = 0.1, susceptibility = 0.1)];
                            background_mortality = 0.01, juvenile_years = 15,
                            juvenile_inoculum = 0.1, regeneration_rate = 0.05, heritability = 0.5)
        sol = solve(HostProblem(prob, host))
        @test all(iszero, sol.fungi)
        @test live_host(sol)[2, 2, end] ≈ 0.5 rtol = 1e-12
        @test live_host(sol; juveniles = false)[2, 2, end] ≈ live_host(sol; juveniles = false)[2, 2, 1] rtol = 1e-12
        @test sol.healthy[2, 2, 2, end] / sol.healthy[2, 2, 1, end] ≈ 0.1 / 0.9 rtol = 1e-12
        @test all(iszero, sol.disease_deaths)
    end

    @testset "accounting and bounds" begin
        prob = uniform_problem()
        host = HostDynamics([HostClass(:a; share = 0.8, mortality = 0.2),
                             HostClass(:b; share = 0.2, susceptibility = 0.2, mortality = 0.02)])
        sol = solve(HostProblem(prob, host))
        @test live_host(sol)[2, 2, :] .+ sol.disease_deaths[2, 2, :] ≈ fill(0.5, length(host_years(sol)))
        full = solve(HostProblem(prob, HostDynamics([HostClass(:a; share = 0.8, mortality = 0.2),
                                                     HostClass(:b; share = 0.2, mortality = 0.02)];
                                                    background_mortality = 0.01, juvenile_years = 10,
                                                    juvenile_mortality_factor = 10.0,
                                                    regeneration_rate = 0.2, heritability = 0.5)))
        @test all(>=(0), full.healthy) && all(>=(0), full.symptomatic)
        @test all(>=(0), full.juvenile_healthy) && all(>=(0), full.juvenile_symptomatic)
        @test maximum(live_host(full)) <= 0.5 + 1e-12
    end

    @testset "selection follows heritability" begin
        tolerant(sol) = sol.healthy[2, 2, 2, end] + sol.symptomatic[2, 2, 2, end] +
                        sol.juvenile_healthy[2, 2, 2, end] + sol.juvenile_symptomatic[2, 2, 2, end]
        run(h) = solve(HostProblem(uniform_problem(; last_year = 2090),
                                   HostDynamics([HostClass(:s; share = 0.95, mortality = 0.3),
                                                 HostClass(:t; share = 0.05, susceptibility = 0.1,
                                                           mortality = 0.01)];
                                                background_mortality = 0.01, juvenile_years = 10,
                                                regeneration_rate = 0.1, heritability = h)))
        fraction(s) = tolerant(s) / live_host(s)[2, 2, end]
        @test fraction(run(1.0)) > fraction(run(0.5)) > fraction(run(0.0)) > 0.05
    end

    @testset "no inoculum means no spores once the imputed years have passed" begin
        sol = solve(HostProblem(uniform_problem(), HostDynamics([HostClass(:a; inoculum = 0.0)])))
        @test all(iszero, sol.fungi[:, :, 3:end])
    end

    @testset "interventions" begin
        ordinary = HostClass(:ash; mortality = 0.1)
        host(v) = HostDynamics([ordinary, v])
        everything = 2001:2030

        @testset "a useless vaccine changes nothing" begin
            prob = uniform_problem(; psi = 0.01)
            plain = solve(HostProblem(prob, host(vaccinated_class(ordinary; efficacy = 0.0))))
            sol = solve(HostProblem(prob, host(vaccinated_class(ordinary; efficacy = 0.0));
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = 2005:2020, rate = 0.3,
                                                               waning = 0.1)]))
            @test sol.fungi ≈ plain.fungi rtol = 1e-12
            @test live_host(sol) ≈ live_host(plain) rtol = 1e-12
            @test sum(sol.applied) > 0
        end

        @testset "a perfect vaccine given to all, with no waning, stops all symptoms" begin
            prob = uniform_problem()
            sol = solve(HostProblem(prob, host(vaccinated_class(ordinary; efficacy = 1.0));
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = everything, rate = 1.0)]))
            @test all(iszero, sol.symptomatic)
            @test all(iszero, sol.disease_deaths)
            @test live_host(sol)[2, 2, end] ≈ 0.5 rtol = 1e-12
            @test sol.applied[1, 1] ≈ 16 * 0.5 rtol = 1e-12
            @test iszero(sum(sol.applied[1, 2:end]))
        end

        @testset "vaccinated trees that make no rachis stop the spores" begin
            sol = solve(HostProblem(uniform_problem(),
                                    host(vaccinated_class(ordinary; efficacy = 0.0, inoculum = 0.0));
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = everything, rate = 1.0)]))
            @test all(iszero, sol.fungi[:, :, 3:end])
        end

        @testset "coverage, the cap and waning follow a hand calculation" begin
            prob = uniform_problem(; psi = 0.0)
            v = vaccinated_class(ordinary; efficacy = 1.0)
            sol = solve(HostProblem(prob, host(v);
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = 2001:2001, rate = 0.5,
                                                               waning = 0.2)]))
            for l in 1:10
                @test sol.healthy[2, 2, 2, l] ≈ 0.25 * 0.8^(l - 1) rtol = 1e-12
                @test sol.healthy[2, 2, 1, l] + sol.healthy[2, 2, 2, l] ≈ 0.5 rtol = 1e-12
            end
            capped = solve(HostProblem(prob, host(v);
                                       interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                                  years = everything, rate = 1.0,
                                                                  max_coverage = 0.3)]))
            @test capped.healthy[2, 2, 2, end] ≈ 0.15 rtol = 1e-12
            @test capped.applied[1, 1] ≈ 16 * 0.15 rtol = 1e-12
            @test iszero(sum(capped.applied[1, 2:end]))
            boosted = solve(HostProblem(prob, host(v);
                                        interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                                   years = everything, rate = 1.0,
                                                                   waning = 0.1)]))
            @test boosted.healthy[2, 2, 2, end] ≈ 0.5 rtol = 1e-12
            @test boosted.applied[1, 5] ≈ 16 * 0.05 rtol = 1e-12
        end

        @testset "symptomatic trees are treated only if asked, and stay symptomatic" begin
            prob = uniform_problem()
            v = vaccinated_class(ordinary; efficacy = 1.0, mortality = 0.0)
            no = solve(HostProblem(prob, host(v);
                                   interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                              years = 2010:2010, rate = 1.0)]))
            @test iszero(no.symptomatic[2, 2, 2, 10])
            yes = solve(HostProblem(prob, host(v);
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = 2010:2010, rate = 1.0,
                                                               symptomatic = true)]))
            @test yes.symptomatic[2, 2, 2, 10] > 0
            @test iszero(yes.symptomatic[2, 2, 1, 10])
            @test yes.symptomatic[2, 2, 2, 11] >= yes.symptomatic[2, 2, 2, 10]
        end

        @testset "seedlings of vaccinated trees are not vaccinated" begin
            prob = uniform_problem(; psi = 0.0)
            sol = solve(HostProblem(prob,
                                    HostDynamics([ordinary, vaccinated_class(ordinary; efficacy = 1.0)];
                                                 background_mortality = 0.01, juvenile_years = 5,
                                                 heritability = 1.0);
                                    interventions = [Vaccinate(; from = :ash, to = :vaccinated_ash,
                                                               years = 2001:2001, rate = 1.0,
                                                               juveniles = false)]))
            @test all(iszero, sol.juvenile_healthy[:, :, 2, :])
            @test sol.juvenile_healthy[2, 2, 1, end] > 0
            @test live_host(sol)[2, 2, end] ≈ 0.5 rtol = 1e-12
        end

        @testset "planting fills gaps, within the window and the cells" begin
            prob = uniform_problem(; last_year = 2060)
            two = HostDynamics([HostClass(:ash; share = 1.0, mortality = 0.5),
                                HostClass(:tolerant; share = 0.0, susceptibility = 0.05)];
                               juvenile_years = 10)
            none = solve(HostProblem(prob, two))
            sol = solve(HostProblem(prob, two;
                                    interventions = [PlantStock(; class = :tolerant, years = 2005:2040,
                                                                rate = 0.05)]))
            @test maximum(live_host(sol)) <= 0.5 + 1e-12
            @test live_host(sol)[2, 2, end] > live_host(none)[2, 2, end]
            @test iszero(sol.applied[1, 1:4])
            @test iszero(sol.applied[1, 41])
            @test sol.applied[1, 10] > 0
            @test sol.applied[1, 10] <= 16 * 0.05 * 0.5 + 1e-12
            outside = solve(HostProblem(prob, two;
                                        interventions = [PlantStock(; class = :tolerant, years = 2005:2040,
                                                                    rate = 0.05, cells = falses(4, 4))]))
            @test live_host(outside) ≈ live_host(none) rtol = 1e-12
            @test iszero(sum(outside.applied))
        end

        @testset "felling removes symptomatic adults and is accounted for" begin
            prob = uniform_problem()
            h = HostDynamics([ordinary])
            sol = solve(HostProblem(prob, h; interventions = [Fell(; years = 2005:2030, rate = 0.2)]))
            live = [sum(live_host(sol)[:, :, l]) for l in axes(sol.healthy, 4)]
            dead = [sum(sol.disease_deaths[:, :, l]) for l in axes(sol.healthy, 4)]
            @test live .+ dead .+ cumsum(sol.applied[1, :]) ≈ fill(16 * 0.5, length(live)) rtol = 1e-12
            none = solve(HostProblem(prob, h))
            @test sol.disease_deaths[2, 2, end] < none.disease_deaths[2, 2, end]
        end

        @testset "litter removal lowers the rachis that makes next year's spores" begin
            prob = uniform_problem(; psi = 0.001)
            e = 0.5
            sol = solve(HostProblem(prob, constant_host();
                                    interventions = [ReduceLitter(; years = 2001:2030, effect = e)]))
            for l in 3:6
                expected = prob.p.beta_0 * (1 - e) * 0.5 * (sol.fungi[2, 2, l - 2] + sol.fungi[2, 2, l - 1])
                @test sol.fungi[2, 2, l] ≈ min(expected, prob.p.S_sat) rtol = 1e-10
            end
            @test live_host(sol)[2, 2, end] ≈ 0.5 rtol = 1e-12
            all_gone = solve(HostProblem(prob, constant_host();
                                         interventions = [ReduceLitter(; years = 2001:2030, effect = 1.0)]))
            @test all(iszero, all_gone.fungi[:, :, 3:end])
        end

        @testset "invalid interventions" begin
            prob = uniform_problem(; psi = 0.0, last_year = 2010)
            kw = (; from = :ash, to = :v, years = 2001:2005)
            @test_throws ArgumentError Vaccinate(; kw..., rate = 1.5)
            @test_throws ArgumentError Vaccinate(; kw..., rate = 0.5, waning = -0.1)
            @test_throws ArgumentError Vaccinate(; kw..., rate = 0.5, max_coverage = 2.0)
            @test_throws ArgumentError Vaccinate(; from = :a, to = :a, years = 2001:2005, rate = 0.5)
            @test_throws ArgumentError PlantStock(; class = :a, years = 2001:2005, rate = -1.0)
            @test_throws ArgumentError Fell(; years = 2001:2005, rate = 2.0)
            @test_throws ArgumentError ReduceLitter(; years = 2001:2005, effect = 2.0)
            @test_throws ArgumentError vaccinated_class(ordinary; efficacy = 1.5)
            @test_throws ArgumentError HostProblem(prob, constant_host();
                                                   interventions = [PlantStock(; class = :none,
                                                                               years = 2001:2005, rate = 0.1)])
            @test_throws ArgumentError HostProblem(prob, constant_host();
                                                   interventions = [Fell(; years = 2001:2005, rate = 0.1,
                                                                         cells = falses(2, 2))])
            @test_throws ArgumentError HostDynamics([HostClass(:a; offspring = :b)])
            @test_throws ArgumentError HostDynamics([HostClass(:a; share = 0.5), HostClass(:a; share = 0.5)])
        end
    end

    @testset "invalid arguments" begin
        @test_throws ArgumentError HostClass(:a; share = 1.5)
        @test_throws ArgumentError HostClass(:a; mortality = -0.1)
        @test_throws ArgumentError HostClass(:a; susceptibility = -1.0)
        @test_throws ArgumentError HostClass(:a; inoculum = -1.0)
        @test_throws ArgumentError HostDynamics(HostClass[])
        @test_throws ArgumentError HostDynamics([HostClass(:a; share = 0.5)])
        @test_throws ArgumentError HostDynamics([HostClass(:a)]; juvenile_years = 0.5)
        @test_throws ArgumentError HostDynamics([HostClass(:a)]; heritability = 2.0)
        @test_throws ArgumentError HostDynamics([HostClass(:a)]; background_mortality = 1.0)
    end
end
