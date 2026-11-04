using LinearAlgebra: issymmetric, eigvals
using SparseArrays: nnz

# The optimised sparse solver is tested against the dense reference of PEOC 1.0,
# elementwise in relative terms: the likelihood reads the tails of the field.
@testset "diffusion" begin
    L = neumann_laplacian(5, 4)
    @test size(L) == (20, 20)
    @test issymmetric(L)
    @test all(iszero, sum(L; dims = 2))
    @test minimum(eigvals(Matrix(L))) ≈ 0 atol = 1e-12
    @test nnz(neumann_laplacian(68, 67)) == 4556 + 2 * (67 * 67 + 68 * 66)

    # a long strip, so that the far field is in the deep tail even at the largest D: its
    # minimum is about 6e-239, 2e-101 and 4e-38 for the three values of D below
    nx, ny = 120, 6
    L = neumann_laplacian(nx, ny)
    s = AshDieback.CrankNicolsonScheme(16.0, 0.1, 60.0, 600)
    nu = zeros(nx * ny)
    nu[1] = 1e4  # a point source at one end of the strip
    nu[2] = 3.0
    for D in (2.0, 30.0, 150.0)
        dense = AshDieback.diffuse(AshDieback.diffusion_operator(DenseCrankNicolson(), L, D, s),
                                   nu, s)
        sparse = AshDieback.diffuse(AshDieback.diffusion_operator(SparseCrankNicolson(), L, D,
                                                                  s), nu, s)
        @test all(>(0), dense)
        @test minimum(dense) < 1e-30
        @test maximum(abs.(sparse ./ dense .- 1)) < 1e-11
    end

    # mass balance: zero-flux boundaries keep all spores, ∑w(τ) = ∑ν
    op = AshDieback.diffusion_operator(SparseCrankNicolson(), L, 20.0, s)
    @test sum(AshDieback.diffuse(op, nu, s)) ≈ sum(nu) rtol = 1e-12

    @test_logs (:warn, r"c = DΔt/\(2h²\)") AshDieback.diffusion_operator(SparseCrankNicolson(),
                                                                      L, 2000.0, s)
end
