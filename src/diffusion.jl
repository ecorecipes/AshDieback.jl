"""
    neumann_laplacian(nx, ny)

The 5-point Laplacian on an `nx × ny` grid with zero-flux (Neumann) boundaries, as a sparse
positive semi-definite matrix L (sign flipped, times h²) acting on `vec` of an `nx × ny`
array: L = I ⊗ L₁(nx) + L₁(ny) ⊗ I with L₁(m) = tridiag(−1, 2, −1) and corner entries 1.
This is the cell-centred, mirror-ghost-cell discretisation that PEOC uses.
"""
function neumann_laplacian(nx::Integer, ny::Integer)
    function L1(m)
        d = fill(2.0, m)
        d[1] = d[end] = 1.0
        return spdiagm(-1 => fill(-1.0, m - 1), 0 => d, 1 => fill(-1.0, m - 1))
    end
    return kron(sparse(1.0I, ny, ny), L1(nx)) + kron(L1(ny), sparse(1.0I, nx, nx))
end

"""
    AbstractDiffusionAlgorithm

Supertype of the methods that solve the spore diffusion step of the model,
∂w/∂t = D Δw + ν/τ with w(0) = 0 and zero-flux boundaries, over the diffusion period τ.
Every method advances the same Crank–Nicolson scheme as PEOC:
(I + cL) wᵏ⁺¹ = (I − cL) wᵏ + Δt ν/τ with c = DΔt/(2h²), for N = τ/Δt steps.
"""
abstract type AbstractDiffusionAlgorithm end

"""
    SparseCrankNicolson(; linsolve = CHOLMODFactorization())

Crank–Nicolson steps with one sparse factorisation of I + cL per likelihood, solved through
LinearSolve.jl with the algorithm `linsolve`.

I + cL is a symmetric M-matrix and I − cL has no negative entries while c ≤ 1/4
(D ≤ h²/(2Δt), 1280 km² day⁻¹ for 16 km quadrats), so no step cancels and every value of
the field keeps its relative accuracy, including the tails down to 1e-134 that the
likelihood reads through log(q). A Cholesky factorisation (the default) or an LU with
diagonal pivots keeps this property; a transform (FFT/DCT) solver does not.
"""
struct SparseCrankNicolson{A} <: AbstractDiffusionAlgorithm
    linsolve::A
end
SparseCrankNicolson(; linsolve = CHOLMODFactorization()) = SparseCrankNicolson(linsolve)

"""
    DenseCrankNicolson()

The reference Crank–Nicolson implementation of PEOC 1.0 (its `'exact'` solver): the dense
inverse M = (I + cL)⁻¹, A = 2M − I and b = Δt Mν/τ, then wᵏ⁺¹ = A wᵏ + b. About 30 times
slower than [`SparseCrankNicolson`](@ref) and kept for testing.
"""
struct DenseCrankNicolson <: AbstractDiffusionAlgorithm end

diffusion_constant(D, s) = D * s.delta_t / (2 * (s.h^2))

function check_diffusion_constant(c, D)
    if 4c > 1
        @warn "D = $D gives c = DΔt/(2h²) = $(round(c; sigdigits = 3)) > 1/4: values far " *
              "from the epidemic front may lose their relative accuracy" maxlog = 1
    end
    return nothing
end

# One diffusion operator, built once per likelihood (it depends only on D) and applied to
# every year's spore production.
struct SparseDiffusionOperator{C,B}
    cache::C
    B::B
end

struct DenseDiffusionOperator
    M::Matrix{Float64}
    A::Matrix{Float64}
end

function diffusion_operator(alg::SparseCrankNicolson, L, D, s)
    c = diffusion_constant(D, s)
    check_diffusion_constant(c, D)
    n = size(L, 1)
    lhs = sparse(1.0I, n, n) + c * L
    cache = init(LinearProblem(lhs, zeros(n)), alg.linsolve)
    return SparseDiffusionOperator(cache, sparse(1.0I, n, n) - c * L)
end

function diffusion_operator(::DenseCrankNicolson, L, D, s)
    c = diffusion_constant(D, s)
    n = size(L, 1)
    M = inv(Matrix(1.0I, n, n) + c * Matrix(L))
    return DenseDiffusionOperator(M, 2M - I)
end

function diffuse(op::SparseDiffusionOperator, nu::AbstractVector, s)
    src = s.delta_t .* nu ./ s.diffusion_period
    w = zero(src)
    rhs = similar(src)
    for _ in 1:nsteps(s)
        mul!(rhs, op.B, w)
        rhs .+= src
        op.cache.b = rhs
        w .= solve!(op.cache).u
    end
    return w
end

function diffuse(op::DenseDiffusionOperator, nu::AbstractVector, s)
    b = s.delta_t .* (op.M * nu) ./ s.diffusion_period
    w = zero(b)
    tmp = similar(b)
    for _ in 1:nsteps(s)
        mul!(tmp, op.A, w)
        w .= tmp .+ b
    end
    return w
end
