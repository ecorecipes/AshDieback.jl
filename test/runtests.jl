using Test
using AshDieback
using TOML

const REFERENCE = TOML.parsefile(joinpath(@__DIR__, "reference", "peoc_reference.toml"))
const DATA = load_peoc_data()

@testset "AshDieback" begin
    include("test_data.jl")
    include("test_diffusion.jl")
    include("test_likelihood.jl")
    include("test_posterior.jl")
    include("test_mcmc.jl")
    include("test_amis.jl")
    include("test_validation.jl")
    include("test_extensions.jl")
    include("test_errors.jl")
    include("test_docstrings.jl")
end
