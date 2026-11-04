@testset "errors" begin
    dir = mktempdir()
    @test_throws DataFormatError load_peoc_data(dir)
    for f in ("climate_data", "density.txt", "data_chalara.txt", "rachis_initialization.txt")
        cp(joinpath(datadir(), f), joinpath(dir, f))
    end
    open(joinpath(dir, "density.txt"), "a") do io
        write(io, "\r1.0 not_a_number 3\r")
    end
    err = try
        load_peoc_data(dir)
    catch e
        e
    end
    @test err isa DataFormatError
    @test occursin("density.txt", sprint(showerror, err))
    @test occursin("not_a_number", sprint(showerror, err))
    @test DataFormatError <: AshDiebackError
    @test DegenerateProposalError <: AshDiebackError

    @test_throws ArgumentError load_peoc_data(; temperature_index = :T29)
    @test_throws ArgumentError AshDiebackParameters(ones(9))
    prob = AshDiebackProblem(DATA, ones(10))
    @test_throws ArgumentError AshDiebackPosterior(prob, ones(9, 2))
    @test_throws ArgumentError AshDiebackPosterior(prob, [ones(10) zeros(10)])
    @test_throws ArgumentError AshDiebackProblem(DATA, ones(10);
                                                 settings = ModelSettings(; final_year = 2030))
end
