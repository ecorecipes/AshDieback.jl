"""
    RMSETable

Output of [`rmse`](@ref). Fields, one entry per survey year: `years`, `observations` (plot
visits), `annual` (RMSE of that year) and `cumulative` (RMSE from the first year to that
year).
"""
struct RMSETable
    years::Vector{Int}
    observations::Vector{Int}
    annual::Vector{Float64}
    cumulative::Vector{Float64}
end

"""
    rmse(sol::AshDiebackSolution)

The root mean square error of the probability of symptoms against the survey (Table 4 of
the paper, PEOC's action 5), for each survey year of `sol` and cumulatively. It compares
every individual plot visit with its quadrat's q, not the per-quadrat means that the
likelihood uses.

To validate on the years after the fit, as Table 4 does, extend the problem first:
`rmse(solve(remake(prob; settings = ModelSettings(; final_year = 2023))))`.
"""
function rmse(sol::AshDiebackSolution)
    survey = sol.prob.survey
    n = length(survey)
    observations, annual, cumulative = zeros(Int, n), zeros(n), zeros(n)
    total, count = 0.0, 0
    for (l, o) in enumerate(survey)
        q = view(sol.symptoms, :, :, l)
        s = sum(abs2(o.visit_proportion[i] - q[o.visit_x[i], o.visit_y[i]])
                for i in eachindex(o.visit_x); init = 0.0)
        m = length(o.visit_x)
        total += s
        count += m
        observations[l] = m
        annual[l] = sqrt(s / m)
        cumulative[l] = sqrt(total / count)
    end
    return RMSETable([o.year for o in survey], observations, annual, cumulative)
end

function Base.show(io::IO, ::MIME"text/plain", t::RMSETable)
    println(io, "RMSETable: year, observations, RMSE of the year, RMSE from ", first(t.years))
    for i in eachindex(t.years)
        println(io, "  ", t.years[i], "  ", lpad(t.observations[i], 4), "  ",
                rpad(round(t.annual[i]; digits = 4), 6, "0"), "  ",
                rpad(round(t.cumulative[i]; digits = 4), 6, "0"))
    end
end

"""
    LikelihoodGrid

Output of [`likelihood_grid`](@ref). Fields: `names` (the varied parameters), `axes` (their
values), `loglikelihood` (an array with one dimension per varied parameter, `-Inf` where the
likelihood is zero) and `prob` (the problem, which gives the parameters held fixed).
"""
struct LikelihoodGrid{N,P<:AshDiebackProblem}
    names::NTuple{N,Symbol}
    axes::NTuple{N,Vector{Float64}}
    loglikelihood::Array{Float64,N}
    prob::P
end

"""
    likelihood_grid(prob::AshDiebackProblem, axes::NamedTuple; alg = SparseCrankNicolson(),
                    threaded = Threads.nthreads() > 1)

The log-likelihood on the Cartesian product of `axes`, which maps parameter names to their
values (PEOC's action 8). The other parameters keep their values in `prob`. For example
`likelihood_grid(prob, (; beta_0 = 20:2:30, D = 16.0:20.0))`. See [`peoc_grid_axes`](@ref)
for PEOC's default grid, and [`profile_likelihood`](@ref) and [`best_parameters`](@ref) for
reading the result.
"""
function likelihood_grid(prob::AshDiebackProblem, axes::NamedTuple;
                         alg::AbstractDiffusionAlgorithm = SparseCrankNicolson(),
                         threaded::Bool = Threads.nthreads() > 1)
    names = keys(axes)
    all(in(parameter_names()), names) ||
        throw(ArgumentError("axes must be parameters in $(parameter_names()), not $names"))
    values = map(v -> Vector{Float64}(collect(v)), Tuple(axes))
    all(!isempty, values) || throw(ArgumentError("every axis must have at least one value"))
    cells = CartesianIndices(length.(values))
    positions = map(n -> findfirst(==(n), parameter_names()), names)
    function evaluate(c)
        p = collect(Float64, prob.p)
        for (d, j) in enumerate(positions)
            p[j] = values[d][c[d]]
        end
        return loglikelihood(remake(prob; p), alg)
    end
    ll = Array{Float64}(undef, length.(values)...)
    if threaded
        Threads.@threads :dynamic for c in cells
            ll[c] = evaluate(c)
        end
    else
        for c in cells
            ll[c] = evaluate(c)
        end
    end
    return LikelihoodGrid(names, values, ll, prob)
end

"""
    peoc_grid_axes()

The regular grid shipped with PEOC 1.0 (`8_parameters_for_computation_on_regular_grid.py`):
β₀, D, S, C_init, γ and κ, 630 000 points, for use with [`likelihood_grid`](@ref).
"""
function peoc_grid_axes()
    return (; beta_0 = range(2, 100; length = 50), D = range(8.0, 35.0; length = 28),
            S_sat = range(80.0, 100.0; length = 3), gamma = range(21.001, 23.001; length = 5),
            C_init = range(0.008, 0.009; length = 3), kappa = range(0.025, 0.25; length = 10))
end

"""
    profile_likelihood(grid::LikelihoodGrid, names::Symbol...)

The grid's log-likelihood maximised over every parameter except `names`, as an array with
one dimension for each of `names`, in that order.
"""
function profile_likelihood(grid::LikelihoodGrid, names::Symbol...)
    all(in(grid.names), names) ||
        throw(ArgumentError("$names are not all axes of the grid, $(grid.names)"))
    allunique(names) || throw(ArgumentError("repeated parameter in $names"))
    keep = map(n -> findfirst(==(n), grid.names), names)
    drop = Tuple(d for d in 1:ndims(grid.loglikelihood) if !(d in keep))
    m = isempty(drop) ? grid.loglikelihood : dropdims(maximum(grid.loglikelihood; dims = drop); dims = drop)
    sorted = Tuple(sortperm(collect(keep)))
    return permutedims(m, invperm(collect(sorted)))
end

"""
    best_parameters(grid::LikelihoodGrid)

The grid point with the highest log-likelihood, as `(parameters, loglikelihood)`.
"""
function best_parameters(grid::LikelihoodGrid)
    c = argmax(grid.loglikelihood)
    p = collect(Float64, grid.prob.p)
    for (d, n) in enumerate(grid.names)
        p[findfirst(==(n), parameter_names())] = grid.axes[d][c[d]]
    end
    return AshDiebackParameters(p), grid.loglikelihood[c]
end
