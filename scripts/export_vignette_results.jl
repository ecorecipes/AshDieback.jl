# Reduce the reference AMIS run (scripts/run_amis.jl, about 25 h on 8 threads) to the small files that
# vignette 5 reads: exact summaries from all its draws, 2000 draws resampled by weight, and its log.
#
#   julia --project scripts/export_vignette_results.jl [run folder]
using AshDieback
using Printf: @printf
using Random: Xoshiro, shuffle
using Serialization: deserialize
using TOML

folder = get(ARGS, 1, joinpath(pkgdir(AshDieback), "runs", "amis_T28"))
out = joinpath(pkgdir(AshDieback), "vignettes", "05_reproducing_the_paper", "results")
mkpath(out)
res, _ = deserialize(joinpath(folder, "checkpoint.jls"))
X, λ, w = amis_draws(res)
names = ["beta_0", "D", "S_sat", "C_init", "kappa"]

m = X * w
Xc = X .- m
Σ = (Xc .* w') * Xc'
sds = sqrt.([Σ[i, i] for i in axes(Σ, 1)])
xbest, λbest = best_sample(res)
summary = Dict("iterations" => length(res.samples), "draws" => size(X, 2),
               "ess" => effective_sample_size(res), "largest_weight_times_average" => maximum(w) * length(w),
               "parameters" => names, "mean" => m, "sd" => sds,
               "q025" => [weighted_quantile(X[j, :], w, 0.025) for j in axes(X, 1)],
               "q975" => [weighted_quantile(X[j, :], w, 0.975) for j in axes(X, 1)],
               "correlation" => [[Σ[i, j] / (sds[i] * sds[j]) for j in axes(Σ, 2)] for i in axes(Σ, 1)],
               "best" => xbest)
open(io -> TOML.print(io, summary), joinpath(out, "amis_reference.toml"), "w")

rng = Xoshiro(1)
n = 2000
c = cumsum(w ./ sum(w))
u0 = rand(rng) / n
picks = shuffle(rng, [min(searchsortedfirst(c, u0 + (i - 1) / n), length(w)) for i in 1:n])
open(joinpath(out, "amis_reference_draws.csv"), "w") do io
    println(io, join(names, ","))
    foreach(k -> println(io, join(X[:, k], ",")), picks)
end

log_lines = readlines(joinpath(folder, "log.csv"))
head = split(log_lines[1], ',')
col(n) = findfirst(==(n), head)
open(joinpath(out, "amis_reference_log.csv"), "w") do io
    println(io, "iteration,seconds,ess,best_logdensity")
    for l in log_lines[2:end]
        r = split(l, ',')
        println(io, r[col("iteration")], ",", r[col("seconds")], ",", r[col("ess")], ",", r[col("best_logdensity")])
    end
end
@printf("%d draws, ESS %.0f, wrote %s\n", size(X, 2), effective_sample_size(res), out)
