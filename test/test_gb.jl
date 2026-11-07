using AshDieback, Test

@testset "Great Britain inputs" begin
    inputs = load_gb_inputs()
    @test length(inputs.squares) == 2827
    @test isapprox(sum(inputs.ash_ha), 284_072; atol = 1)
    @test count(inputs.seeded) == 734
    @test inputs.t28_years == 2004:2025
    @test size(inputs.t28) == (2827, 22)

    future = load_gb_future(inputs)
    @test length(future) == 8
    @test all(m -> size(m) == (2827, 55) && all(>=(0), m), values(future))
    w = inputs.ash_ha ./ sum(inputs.ash_ha)
    early = sum(w .* vec(sum(future[("rcp85", "01")][:, 1:10]; dims = 2))) / 10
    late = sum(w .* vec(sum(future[("rcp85", "01")][:, 46:55]; dims = 2))) / 10
    @test late > early

    B = 25.0
    prob, cell = gb_problem(inputs; B, final_year = 2026)
    @test size(prob.data.ash_density) == size(prob.data.grid)
    @test isapprox(sum(prob.data.ash_density), sum(inputs.ash_ha) / 100 / 100 * B; rtol = 1e-12)
    k = findfirst(>(0), inputs.ash_ha)
    @test prob.data.ash_density[cell(inputs.squares[k]...)...] ≈ inputs.ash_ha[k] / 100 / 100 * B
    @test prob.data.temperature[cell(inputs.squares[k]...)..., 1] == inputs.t28[k, 1]

    # A later temperature replaces the resampling, and leaves observed years alone.
    warm, _ = gb_problem(inputs; B, temperature = future[("rcp85", "01")], last_year = 2100)
    cool, _ = gb_problem(inputs; B)
    @test warm.data.temperature[:, :, 1:22] == cool.data.temperature[:, :, 1:22]
    i, j = cell(inputs.squares[k]...)
    @test warm.data.temperature[i, j, 2030 - 2003] == future[("rcp85", "01")][k, 2030 - 2025]

    # The share symptomatic in 2026 under the central host, from the sweeps' own code.
    sus = HostClass(:susceptible; share = 0.95, mortality = 0.04)
    tol = HostClass(:tolerant; share = 0.05, susceptibility = 0.2, mortality = 0.004)
    host = HostDynamics([sus, tol]; background_mortality = 0.005, juvenile_years = 15.0,
                        juvenile_mortality_factor = 2.5, juvenile_inoculum = 0.1,
                        regeneration_rate = 0.03, heritability = 0.5)
    sol = solve(HostProblem(prob, host))
    K = prob.data.ash_density
    use = K .> 0
    q = view(symptom_share(sol), :, :, 2026 - first(host_years(sol)) + 1)
    @test sum(K[use] .* q[use]) / sum(K[use]) ≈ 0.5331147047996979 atol = 1e-9
end
