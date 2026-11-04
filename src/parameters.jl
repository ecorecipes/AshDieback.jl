"""
    AshDiebackParameters(; r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_pers, C_init,
                         kappa)
    AshDiebackParameters(v::AbstractVector)

The ten model parameters, in PEOC's order. It is an `AbstractVector`, so `θ[5] == θ.D`
and `collect(θ)` gives the vector that PEOC saves.

| index | field    | paper  | meaning                                              |
|:----- |:-------- |:------ |:---------------------------------------------------- |
| 1     | `r_symp` | r_S    | scale of the symptom probability                     |
| 2     | `r_allee`| r      | Allee threshold of spore production                  |
| 3     | `beta_0` | β₀     | baseline spore production                            |
| 4     | `beta_1` | β₁     | effect of June rainfall on spore production          |
| 5     | `D`      | D      | diffusion coefficient (km² day⁻¹)                    |
| 6     | `S_sat`  | S      | saturation of the spore load                         |
| 7     | `gamma`  | γ      | temperature index that stops symptoms                |
| 8     | `C_pers` | C_pers | probability that symptoms persist (`C_non_recover`)  |
| 9     | `C_init` | C_init | scale of the imputed initial prevalence              |
| 10    | `kappa`  | κ      | exponent of the temperature function (`puiss`)       |

Scaling `(r_symp, r_allee, S_sat)` by the same factor leaves the model unchanged, so the
paper fixes `r_symp = 1000`.
"""
struct AshDiebackParameters{T<:Real} <: AbstractVector{T}
    r_symp::T
    r_allee::T
    beta_0::T
    beta_1::T
    D::T
    S_sat::T
    gamma::T
    C_pers::T
    C_init::T
    kappa::T
end

function AshDiebackParameters(; r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma, C_pers,
                              C_init, kappa)
    return AshDiebackParameters(promote(r_symp, r_allee, beta_0, beta_1, D, S_sat, gamma,
                                        C_pers, C_init, kappa)...)
end

function AshDiebackParameters(v::AbstractVector{<:Real})
    length(v) == 10 ||
        throw(ArgumentError("a parameter vector has 10 entries, not $(length(v))"))
    return AshDiebackParameters(promote(v...)...)
end

AshDiebackParameters(θ::AshDiebackParameters) = θ

Base.size(::AshDiebackParameters) = (10,)
Base.getindex(θ::AshDiebackParameters, i::Int) = getfield(θ, i)
Base.IndexStyle(::Type{<:AshDiebackParameters}) = IndexLinear()

"""
    parameter_names()

The names of the ten parameters as `Symbol`s, in the order of
[`AshDiebackParameters`](@ref).
"""
parameter_names() = fieldnames(AshDiebackParameters)

"""
    temperature_effect(T, gamma, kappa)

The temperature function f(T) = ([1 − T/γ]₊)^κ of the symptom model (eq. 7 of the paper):
symptoms become rarer as summers get hotter and stop at T ≥ γ. Follows IEEE powers:
`0^κ = 0` for κ > 0 and `0^0 = 1`.
"""
temperature_effect(T::Real, gamma::Real, kappa::Real) = max(1 - T * 1.0 / gamma, 0)^kappa
