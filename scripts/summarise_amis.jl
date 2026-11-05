# Compare an AMIS run from scripts/run_amis.jl with the paper: θmax (eq. 14), the correlations
# (Table 3), and Table 4 at the best sample.
#
#   julia --project scripts/summarise_amis.jl [folder]
using AshDieback
using LinearAlgebra: Diagonal
using Printf: @printf
using Serialization: deserialize

folder = get(ARGS, 1, joinpath(pkgdir(AshDieback), "runs", "amis_T28"))
res, _ = deserialize(joinpath(folder, "checkpoint.jls"))
θ0 = AshDiebackParameters(; r_symp = 1000.0, r_allee = 0.0, beta_0 = 24.0, beta_1 = 0.0,
                          D = 18.0, S_sat = 90.0, gamma = 21.50001, C_pers = 1.0,
                          C_init = 0.0085, kappa = 0.05)
prob = AshDiebackProblem(load_peoc_data(), θ0)
post = AshDiebackPosterior(prob, reduced_model_prior_ranges())
names = free_parameters(post)
X, λ, w = amis_draws(res)
@printf("%d iterations, %d draws, %d with zero density, ESS %.1f\n", length(res.samples),
        size(X, 2), count(!isfinite, λ), effective_sample_size(res))

θmax = (beta_0 = 23.26, D = 18.51, S_sat = 92.0, C_init = 0.00855, kappa = 0.056)
xbest, λbest = best_sample(res)
lprior = logprior(post, xbest)
@printf("\n%-8s %10s %10s | %10s %10s %10s %10s %10s\n", "", "θmax", "best", "mean", "sd", "q2.5",
        "q50", "q97.5")
m = X * w
for (j, n) in enumerate(names)
    @printf("%-8s %10.5g %10.5g | %10.5g %10.3g %10.5g %10.5g %10.5g\n", n, θmax[n], xbest[j], m[j],
            sqrt(sum(w .* (X[j, :] .- m[j]) .^ 2)), weighted_quantile(X[j, :], w, 0.025),
            weighted_quantile(X[j, :], w, 0.5), weighted_quantile(X[j, :], w, 0.975))
end
@printf("\nbest log-likelihood %.3f; at θmax %.3f (paper's parameter set)\n", λbest - lprior,
        loglikelihood(remake(prob; p = [1000.0, 0.0, 23.26, 0.0, 18.51, 92.0, 21.50001, 1.0,
                                        0.00855, 0.056])))

Σ = res.covariances[end]
D = Diagonal(sqrt.(1 ./ [Σ[i, i] for i in axes(Σ, 1)]))
R = D * Σ * D
order = [2, 1, 5, 3, 4]  # D, β₀, κ, S, C_init: the order of Table 3
labels = ["D", "β0", "κ", "S", "Cinit"]
paper = [1.0 NaN NaN NaN NaN; -0.97 1.0 NaN NaN NaN; 0.06 -0.04 1.0 NaN NaN
         0.09 -0.11 0.58 1.0 NaN; -0.09 0.08 0.02 -0.14 1.0]
println("\nCorrelations: this run (paper's Table 3)")
for i in 2:5
    print(rpad(labels[i], 7))
    for j in 1:(i - 1)
        @printf("  %6.2f (%5.2f)", R[order[i], order[j]], paper[i, j])
    end
    println()
end

println("\nTable 4 at the best sample of this run")
extended = remake(prob; p = full_parameters(post, xbest),
                  settings = ModelSettings(; final_year = 2023))
display(rmse(solve(extended)))
