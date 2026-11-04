# numpy's log: NaN (not a DomainError) for negative arguments
nanlog(x::Float64) = x < 0 ? NaN : log(x)

"""
    peoc_loglikelihood(sol::AshDiebackSolution)

The log-likelihood exactly as PEOC computes it: a pair `(ℓ, zero)` where `zero` is true
when the likelihood is exactly zero (a quadrat with q = 0 and an infected tree, or q = 1
and a healthy one). PEOC stores `ℓ * (1 - zero)`, so a stored 0 means a zero likelihood.

PEOC's quirks are kept: quadrats with q = 1 *reset* the flag of the same year instead of
adding to it, the sum stops at the first flagged year but keeps its partial value, and a
probability outside [0, 1] gives `NaN`. [`loglikelihood`](@ref) is the corrected version.
"""
function peoc_loglikelihood(sol::AshDiebackSolution)
    prob = sol.prob
    ll = 0.0
    zero = false
    for (l, (o, b)) in enumerate(zip(prob.survey, prob.binomial))
        zero && break
        q = view(sol.symptoms, :, :, l)
        qi = [q[o.x[i], o.y[i]] for i in eachindex(o.x)]
        keep = trues(length(qi))
        if any(==(0), qi)
            zero = any(b.infected[qi .== 0] .!= 0)
            keep .&= qi .!= 0
        end
        if any(==(1), qi[keep])
            zero = any(b.healthy[keep .& (qi .== 1)] .!= 0)
            keep .&= qi .!= 1
        end
        ll += sum(b.log_choose[i] + b.infected[i] * nanlog(qi[i]) +
                  b.healthy[i] * nanlog(1 - qi[i]) for i in findall(keep); init = 0.0)
    end
    return ll, zero
end

"""
    loglikelihood(sol::AshDiebackSolution)
    loglikelihood(prob::AshDiebackProblem, alg = SparseCrankNicolson())

The binomial log-likelihood of the survey: the number of trees with symptoms in quadrat i
in year a is Binomial(m × visits, q_a^i), independently. It is `-Inf` when the likelihood is
zero and when a probability falls outside [0, 1] (possible because PEOC does not cap the
symptom probability; see [`ModelSettings`](@ref)).

Equal to PEOC's value except in the corner cases listed in [`peoc_loglikelihood`](@ref).
"""
function loglikelihood(sol::AshDiebackSolution)
    prob = sol.prob
    ll = 0.0
    for (l, (o, b)) in enumerate(zip(prob.survey, prob.binomial))
        q = view(sol.symptoms, :, :, l)
        for i in eachindex(o.x)
            qi = q[o.x[i], o.y[i]]
            (0 <= qi <= 1) || return -Inf
            k, nk = b.infected[i], b.healthy[i]
            (qi == 0 && k != 0) && return -Inf
            (qi == 1 && nk != 0) && return -Inf
            ll += b.log_choose[i] + (k == 0 ? 0.0 : k * log(qi)) +
                  (nk == 0 ? 0.0 : nk * log1p(-qi))
        end
    end
    return ll
end

loglikelihood(prob::AshDiebackProblem, alg::AbstractDiffusionAlgorithm = SparseCrankNicolson()) =
    loglikelihood(solve(prob, alg))
