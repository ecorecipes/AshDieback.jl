# The data layer must reproduce PEOC's arrays exactly (bitwise): probes are compared with
# ==, sums to rounding (summation order differs).
@testset "data" begin
    ref = REFERENCE["data"]
    @test size(DATA.grid) == (68, 67)
    @test ncells(DATA.grid) == 4556
    @test DATA.climate_years == 2007:2022
    arrays = Dict("H" => DATA.rainfall, "T" => DATA.temperature,
                  "ash_density" => DATA.ash_density, "infection_1" => DATA.psi[1],
                  "infection_2" => DATA.psi[2])
    for (name, A) in arrays
        @testset "$name" begin
            @test count(!=(0), A) == ref["$(name)_nonzero"]
            @test sum(A) ≈ ref["$(name)_sum"] rtol = 1e-13
            for p in ref["$(name)_probes"]
                @test A[Int.(p[1:(end - 1)])...] == p[end]
            end
        end
    end

    obs = REFERENCE["observations"]
    @test [o.year for o in DATA.observations] == obs["years"]
    @test length.(DATA.observations) == obs["quadrats"]
    @test [sum(o.visits) for o in DATA.observations] == obs["visits"]
    @test [sum(o.proportion) for o in DATA.observations] ≈ obs["proportion_sum"] rtol = 1e-13
    first2008 = obs["first_2008"]
    o = DATA.observations[1]
    @test (o.x[1], o.y[1], o.proportion[1], o.visits[1]) ==
          (first2008[1], first2008[2], first2008[3], first2008[4])
    terms = [BinomialTerms(o, 30) for o in DATA.observations]
    @test [sum(t.log_choose) for t in terms] ≈ obs["log_choose_sum"] rtol = 1e-12
    @test all(t -> all(isinteger, t.infected) && all(>=(0), t.healthy), terms)

    @testset "grid" begin
        g = QuadratGrid()
        @test quadrat_index(g, 7_000.0, 1_660_000.0) == (1, 1)
        @test quadrat_index(g, 7_000.0 + 8_000, 1_660_000.0) == (1, 1)  # tie to even
        @test quadrat_index(g, 7_000.0 + 24_000, 1_660_000.0) == (3, 1)
        @test quadrat_index(g, 1_047_000.0, 1_700_000.0) === nothing  # Corsica
    end

    @testset "temperature index" begin
        d24 = load_peoc_data(; temperature_index = :T24)
        @test d24.temperature != DATA.temperature
        @test d24.rainfall == DATA.rainfall
    end
end
