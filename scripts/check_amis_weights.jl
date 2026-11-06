# How much do the largest weights of an AMIS run matter? The proposals are Gaussian and the posterior's
# tails along the β₀–D ridge are heavier, so one draw can take a large share and lower the effective
# sample size sharply. This prints the largest weights, and the run's summaries without the largest.
#
#   julia --project scripts/check_amis_weights.jl [folder]
using AshDieback
using Printf: @printf
using Serialization: deserialize

folder = get(ARGS, 1, joinpath(pkgdir(AshDieback), "runs", "amis_T28"))
res, _ = deserialize(joinpath(folder, "checkpoint.jls"))
θ0 = AshDiebackParameters(; r_symp = 1000.0, r_allee = 0.0, beta_0 = 24.0, beta_1 = 0.0,
                          D = 18.0, S_sat = 90.0, gamma = 21.50001, C_pers = 1.0,
                          C_init = 0.0085, kappa = 0.05)
post = AshDiebackPosterior(AshDiebackProblem(load_peoc_data(), θ0), reduced_model_prior_ranges())
names = free_parameters(post)
X, λ, w = amis_draws(res)
batch = size(res.samples[1], 2)

function moments(X, w)
    m = X * w
    Xc = X .- m
    Σ = (Xc .* w') * Xc'
    return (; m, sd = sqrt.([Σ[j, j] for j in axes(Σ, 1)]), Σ, ess = 1 / sum(abs2, w))
end

order = sortperm(w; rev = true)
all_draws = moments(X, w)
keep = order[2:end]
without = moments(X[:, keep], w[keep] ./ sum(w[keep]))
δ(k) = X[:, k] .- without.m
mahalanobis(k) = sqrt(δ(k)' * (without.Σ \ δ(k)))

@printf("%d draws; ESS %.0f, or %.0f without the largest weight\n", length(w), all_draws.ess, without.ess)
@printf("best log density %.3f\n\n", maximum(λ[isfinite.(λ)]))
println("Largest weights; x in the order ", names)
for k in order[1:5]
    @printf("iteration %3d: share %.5f (%6.0f x average), distance %.2f, log density %.3f, x = %s\n",
            (k - 1) ÷ batch + 1, w[k], w[k] * length(w), mahalanobis(k), λ[k],
            string(round.(X[:, k]; sigdigits = 5)))
end

println("\nWithout the largest weight, the change in each mean, in sd:")
for (j, n) in enumerate(names)
    @printf("%-8s mean %.6g, sd %.4g; without: mean %.6g, sd %.4g; change %.3f sd\n", n, all_draws.m[j],
            all_draws.sd[j], without.m[j], without.sd[j], (all_draws.m[j] - without.m[j]) / without.sd[j])
end
r(s) = s.Σ[1, 2] / (s.sd[1] * s.sd[2])
@printf("r(β₀, D) %.4f; without %.4f\n", r(all_draws), r(without))
