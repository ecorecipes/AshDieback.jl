using Distributions: MvNormal, logpdf
using LinearAlgebra: Diagonal
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

struct GaussianTarget
    d::MvNormal
end
LogDensityProblems.logdensity(t::GaussianTarget, x) = logpdf(t.d, x)
LogDensityProblems.dimension(t::GaussianTarget) = length(t.d)
LogDensityProblems.capabilities(::Type{GaussianTarget}) = LogDensityProblems.LogDensityOrder{0}()

@testset "amis" begin
    target = GaussianTarget(MvNormal([1.0, -2.0], [1.0 0.6; 0.6 2.0]))
    res = amis(Xoshiro(3), target, [0.0, 0.0], Matrix(Diagonal([9.0, 9.0]));
               iterations = 6, samples = 2_000, threaded = false)
    @test length(res.samples) == 6
    @test length(res.means) == 7
    @test sum(sum, res.weights) ≈ 1
    @test res.means[end] ≈ [1.0, -2.0] atol = 0.1
    @test res.covariances[end] ≈ [1.0 0.6; 0.6 2.0] atol = 0.25
    # threads give the same result: the draws are made before the parallel evaluation
    rt = amis(Xoshiro(3), target, [0.0, 0.0], Matrix(Diagonal([9.0, 9.0]));
              iterations = 2, samples = 500, threaded = true)
    rs = amis(Xoshiro(3), target, [0.0, 0.0], Matrix(Diagonal([9.0, 9.0]));
              iterations = 2, samples = 500, threaded = false)
    @test rt.means == rs.means
    # fewer draws than dimensions: the weighted covariance is singular
    @test_throws DegenerateProposalError amis(Xoshiro(1), target, [0.0, 0.0],
                                              Matrix(Diagonal([1.0, 1.0]));
                                              iterations = 3, samples = 1, threaded = false)
end
