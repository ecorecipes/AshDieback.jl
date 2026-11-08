# The paper's AMIS estimation of the reduced model (Section 5, Table 2): 60 iterations of
# 10 000 draws, T28 index, fitted to 2008-2019. It takes about 24 h on 8 threads, saves a
# checkpoint after every iteration, and resumes from one if it exists.
#
#   julia --project --threads=8 --gcthreads=1 scripts/run_amis.jl [folder] [iterations] [samples] [seed] [delta_t]
#       [cap_exponent] [beta_0 D S_sat C_init kappa]
#
# delta_t defaults to the paper's 0.1. A coarser step is an approximation: delta_t = 1 is 10 times
# faster, with log-likelihood errors of about 0.04 (sd) between draws near the posterior.
# cap_exponent (default 0, the paper's hard cap) is the broken power law above the spore cap,
# `ModelSettings(; cap_exponent)`, held fixed. The five values after it centre the first proposal;
# they default to the paper's region, which is far from the posterior when cap_exponent > 0, so pass
# a fit at that exponent (vignette 6's `broken_refits.csv`).
# Default folder: runs/amis_T28 (git-ignored). Writes checkpoint.jls, which holds the
# AMISResult and the RNG state, and log.csv, one row per iteration. The deviation columns are
# eq. (18) of the paper for each parameter, with cells of a fifth of Table 2's standard
# deviation.
using AshDieback
using Dates: now
using LinearAlgebra: BLAS, Diagonal
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro
using Serialization: deserialize, serialize

BLAS.set_num_threads(1)
folder = get(ARGS, 1, joinpath(pkgdir(AshDieback), "runs", "amis_T28"))
iterations = parse(Int, get(ARGS, 2, "60"))
samples = parse(Int, get(ARGS, 3, "10000"))
seed = parse(Int, get(ARGS, 4, "1"))
delta_t = parse(Float64, get(ARGS, 5, "0.1"))
cap_exponent = parse(Float64, get(ARGS, 6, "0.0"))
length(ARGS) in (0:6..., 11) || error("give all five start values (beta_0 D S_sat C_init kappa) or none")
start = length(ARGS) == 11 ? parse.(Float64, ARGS[7:11]) : [24.0, 18.0, 90.0, 0.0085, 0.05]
mkpath(folder)
checkpoint = joinpath(folder, "checkpoint.jls")
logfile = joinpath(folder, "log.csv")

θ0 = AshDiebackParameters(; r_symp = 1000.0, r_allee = 0.0, beta_0 = start[1], beta_1 = 0.0,
                          D = start[2], S_sat = start[3], gamma = 21.50001, C_pers = 1.0,
                          C_init = start[4], kappa = start[5])
post = AshDiebackPosterior(AshDiebackProblem(load_peoc_data(), θ0;
                                             settings = ModelSettings(; delta_t, cap_exponent)),
                           reduced_model_prior_ranges())
names = free_parameters(post)
@assert names == [:beta_0, :D, :S_sat, :C_init, :kappa]
m1 = collect(θ0)[post.free]
Σ1 = [3.0, 0.4, 1.0, 1e-8, 1.5e-5]  # Table 2, as variances
widths = sqrt.(Σ1) ./ 5

function marginals(res)
    X, _, w = amis_draws(res)
    return map(axes(X, 1)) do j
        cells = Dict{Int,Float64}()
        for i in axes(X, 2)
            w[i] > 0 || continue
            c = floor(Int, X[j, i] / widths[j])
            cells[c] = get(cells, c, 0.0) + w[i]
        end
        cells
    end
end

deviation(a, b) = maximum(abs(get(a, c, 0.0) - get(b, c, 0.0)) for c in union(keys(a), keys(b)))

if isfile(checkpoint)
    res, rng = deserialize(checkpoint)
    println("resuming after iteration ", length(res.samples))
else
    res = AMISResult(Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[], [m1],
                     [Matrix{Float64}(Diagonal(Σ1))])
    rng = Xoshiro(seed)
    open(logfile, "w") do io
        println(io, "iteration,finished,seconds,zero_density,ess,best_logdensity,",
                join(("mean_$n" for n in names), ","), ",",
                join(("deviation_$n" for n in names), ","))
    end
end
remaining = iterations - length(res.samples)
remaining > 0 || (println("already has $(length(res.samples)) iterations"); exit())

previous = isempty(res.samples) ? nothing : marginals(res)
started = Ref(time())
function record(r, k)
    global previous
    tmp = checkpoint * ".tmp"
    serialize(tmp, (r, rng))
    mv(tmp, checkpoint; force = true)
    current = marginals(r)
    devs = previous === nothing ? fill(NaN, length(names)) : deviation.(previous, current)
    previous = current
    open(logfile, "a") do io
        println(io, join((k, now(), round(time() - started[]; digits = 1),
                          count(!isfinite, r.logdensity[end]),
                          round(effective_sample_size(r); digits = 1), best_sample(r)[2],
                          r.means[end]..., devs...), ","))
    end
    println(now(), " iteration ", k, ": ESS ", round(effective_sample_size(r); digits = 1),
            ", best ", best_sample(r)[2], ", ", round(time() - started[]; digits = 1), " s")
    flush(stdout)
    started[] = time()
end

println(now(), " ", Threads.nthreads(), " threads; ", remaining, " iterations of ", samples,
        "; cap_exponent ", cap_exponent)
flush(stdout)
amis(rng, post, res; iterations = remaining, samples, threaded = true, callback = record)
