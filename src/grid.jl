"""
    QuadratGrid(; quadrat_length = 16.0, nx_mainland = 62, ny_mainland = 61,
                border = (3, 3, 3, 3), x_origin = 55_000.0, y_origin = 1_708_000.0,
                x_limit = 1_047_000.0)

The regular grid of square quadrats covering mainland France on which the model runs.

Quadrats are located by extended Lambert II coordinates in metres. The mainland covers
`nx_mainland × ny_mainland` quadrats of side `quadrat_length` km whose south-west corner
quadrat is centred at `(x_origin, y_origin)`. `border = (left, right, bottom, top)` adds
fictitious quadrats with no ash on each side, which moves the zero-flux boundary of the
diffusion away from the coast. Points with `x ≥ x_limit` (Corsica) are not on the grid.

Arrays on the grid have size `size(grid) = (nx, ny)` and are indexed `[x, y]`, west to east
and south to north. See [`quadrat_index`](@ref).
"""
Base.@kwdef struct QuadratGrid
    quadrat_length::Float64 = 16.0
    nx_mainland::Int = 62
    ny_mainland::Int = 61
    border::NTuple{4,Int} = (3, 3, 3, 3)
    x_origin::Float64 = 55_000.0
    y_origin::Float64 = 1_708_000.0
    x_limit::Float64 = 1_047_000.0
end

Base.size(g::QuadratGrid) = (g.nx_mainland + g.border[1] + g.border[2],
                             g.ny_mainland + g.border[3] + g.border[4])
Base.size(g::QuadratGrid, d::Integer) = size(g)[d]

"""
    ncells(grid::QuadratGrid)

Number of quadrats on the grid, borders included (`prod(size(grid))`).
"""
ncells(g::QuadratGrid) = prod(size(g))

"""
    quadrat_index(grid::QuadratGrid, x, y)

The `(i, j)` index (1-based) of the quadrat containing the point with extended Lambert II
coordinates `(x, y)` in metres, or `nothing` if the point is in Corsica (`x ≥ x_limit`).

Coordinates are rounded to the nearest quadrat centre with ties to even, as `numpy.round`
does in PEOC, so the indexing of the input files is reproduced exactly.
"""
function quadrat_index(g::QuadratGrid, x::Real, y::Real)
    x < g.x_limit || return nothing
    L = 1000 * g.quadrat_length
    xmin = g.x_origin - L * g.border[1]
    ymin = g.y_origin - L * g.border[3]
    return (round(Int, (x - xmin) / L) + 1, round(Int, (y - ymin) / L) + 1)
end
