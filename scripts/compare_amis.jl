# Compare a fast AMIS run (coarser delta_t, fewer draws) with a reference run from
# scripts/run_amis.jl, and check the fast draws against the likelihood at delta_t = 0.1.
#
#   julia --project --threads=4 scripts/compare_amis.jl fast_folder reference_folder [delta_t] [n] [cap_exponent]
#
# `n` fast draws (default 400) are resampled by weight and their likelihood recomputed at
# delta_t = 0.1. If the fast run is sound, the correction weights are nearly equal. Pass the
# runs' fixed `cap_exponent` (default 0, the paper's hard cap) so the likelihoods match them.
using AshDieback
using LinearAlgebra: BLAS, Symmetric
using Printf: @printf
using Random: Xoshiro
using Serialization: deserialize
using Statistics: mean, median, std

BLAS.set_num_threads(1)
fast, _ = deserialize(joinpath(ARGS[1], "checkpoint.jls"))
ref, _ = deserialize(joinpath(ARGS[2], "checkpoint.jls"))
delta_t = parse(Float64, get(ARGS, 3, "1.0"))
n = parse(Int, get(ARGS, 4, "400"))
cap_exponent = parse(Float64, get(ARGS, 5, "0.0"))

θ0 = [1000.0, 0.0, 24.0, 0.0, 18.0, 90.0, 21.50001, 1.0, 0.0085, 0.05]
data = load_peoc_data()
exact = AshDiebackPosterior(AshDiebackProblem(data, θ0; settings = ModelSettings(; cap_exponent)),
                            reduced_model_prior_ranges())
coarse = AshDiebackPosterior(AshDiebackProblem(data, θ0; settings = ModelSettings(; delta_t, cap_exponent)),
                             reduced_model_prior_ranges())
names = free_parameters(exact)

function summary(res)
    X, _, w = amis_draws(res)
    m = X * w
    Xc = X .- m
    Σ = (Xc .* w') * Xc'
    q = [[weighted_quantile(X[j, :], w, p) for p in (0.025, 0.5, 0.975)] for j in axes(X, 1)]
    return (; m, sd = sqrt.([Σ[j, j] for j in axes(Σ, 1)]), q, Σ)
end

a, b = summary(fast), summary(ref)
@printf("fast:      %d iterations, %d draws, ESS %.0f\n", length(fast.samples),
        sum(size.(fast.samples, 2)), effective_sample_size(fast))
@printf("reference: %d iterations, %d draws, ESS %.0f\n\n", length(ref.samples),
        sum(size.(ref.samples, 2)), effective_sample_size(ref))
@printf("%-8s %10s %10s %8s | %10s %10s | %10s %10s %10s %10s\n", "", "mean", "ref mean", "Δ/sd",
        "sd", "ref sd", "q2.5", "ref q2.5", "q97.5", "ref q97.5")
for (j, name) in enumerate(names)
    @printf("%-8s %10.5g %10.5g %8.3f | %10.3g %10.3g | %10.5g %10.5g %10.5g %10.5g\n", name, a.m[j],
            b.m[j], (a.m[j] - b.m[j]) / b.sd[j], a.sd[j], b.sd[j], a.q[j][1], b.q[j][1], a.q[j][3],
            b.q[j][3])
end

corr(s) = [s.Σ[i, j] / (s.sd[i] * s.sd[j]) for i in axes(s.Σ, 1), j in axes(s.Σ, 2)]
println("\nCorrelations, fast (reference):")
R, Q = corr(a), corr(b)
for i in 2:length(names)
    print(rpad(names[i], 8))
    for j in 1:(i - 1)
        @printf("  %6.2f (%5.2f)", R[i, j], Q[i, j])
    end
    println()
end

# the fast draws against the likelihood at delta_t = 0.1
X, λ, w = amis_draws(fast)
rng = Xoshiro(1)
cw = cumsum(w)
picks = [searchsortedfirst(cw, min(rand(rng), cw[end])) for _ in 1:n]
ex = zeros(n)
co = zeros(n)
Threads.@threads :dynamic for k in 1:n
    x = view(X, :, picks[k])
    ex[k] = loglikelihood(exact, x)
    co[k] = loglikelihood(coarse, x)
end
d = ex .- co
c = exp.(d .- maximum(d))
@printf("\n%d fast draws, resampled by weight, at delta_t = 0.1 against %g:\n", n, delta_t)
@printf("  log-likelihood shift: median %.4f, sd %.4f, range %.4f to %.4f\n", median(d), std(d),
        extrema(d)...)
@printf("  efficiency of the correction weights: %.4f\n", sum(c)^2 / (n * sum(abs2, c)))
xb = best_sample(fast)[1]
@printf("  best fast draw at delta_t = 0.1: %.3f%s\n", loglikelihood(exact, xb),
        cap_exponent == 0 ? " (θmax -24717.215)" : "")
xr = best_sample(ref)[1]
@printf("  best reference draw: %.3f\n", loglikelihood(exact, xr))
