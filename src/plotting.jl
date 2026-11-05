# Plot recipes (RecipesBase): `plot(sol; year, field)` with Plots.jl loaded.

@recipe function f(sol::AshDiebackSolution; year = nothing, field = :symptoms)
    field in (:symptoms, :fungi) ||
        throw(ArgumentError("field must be :symptoms or :fungi, not $(repr(field))"))
    years = field === :symptoms ? survey_years(sol.prob.settings) :
            simulated_years(sol.prob.settings)
    a = year === nothing ? last(years) : year
    a in years || throw(ArgumentError("year $a is not in $years"))
    Z = (field === :symptoms ? sol.symptoms : sol.fungi)[:, :, a - first(years) + 1]
    Z = copy(Z)
    Z[sol.prob.data.ash_density .== 0] .= NaN  # no ash: outside France or no host
    seriestype := :heatmap
    aspect_ratio --> :equal
    title --> (field === :symptoms ? "probability of symptoms, $a" :
               "colonised rachis per ash, $a")
    colorbar_title --> (field === :symptoms ? "q" : "χ/d")
    xlabel --> "x quadrat"
    ylabel --> "y quadrat"
    return permutedims(Z)
end

# `plot(grid; x = :beta_0, y = :D)`: the profile log-likelihood over two parameters
@recipe function f(grid::LikelihoodGrid; x = grid.names[1], y = grid.names[2])
    Z = profile_likelihood(grid, y, x)
    seriestype := :heatmap
    xlabel --> string(x)
    ylabel --> string(y)
    colorbar_title --> "profile log-likelihood"
    return grid.axes[findfirst(==(x), grid.names)], grid.axes[findfirst(==(y), grid.names)], Z
end
