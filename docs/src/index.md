# AshDieback.jl

```@meta
CurrentModule = AshDieback
```

```@docs
AshDieback
```

AshDieback.jl implements the mechanistic-statistical model of the spread of ash dieback
(*Hymenoscyphus fraxineus*, "Chalara") across France of Fritsch, Grosdidier,
Gégout-Petit and Marçais, *Mechanistic-statistical model for the expansion of ash
dieback* ([HAL 04690647](https://hal.science/hal-04690647)). It is a port of PEOC, the
authors' Python code, and is tested against it.

## The model

The model runs on 16 km quadrats with a yearly step and has three nested parts.

1. **Rachis colonisation** (deterministic). Leaf debris (rachises) colonised in the two
   previous years, R_a = χ_{a−1} + χ_{a−2}, produce spores ν_a = H_a R_a above an Allee
   threshold r and H_a R_a²/r below it, with H_a = β₀ + β₁ h_a and h_a the June rainfall.
   The spores diffuse for τ = 60 days, ∂w/∂t = D Δw + ν_a/τ with zero-flux boundaries,
   and the newly colonised rachis is χ_a = min(w(τ), S) d, with d the ash density.
2. **Symptoms** (Bernoulli per tree). A tree develops symptoms in year a with probability
   q̃_a = (χ_{a−1}/d) f(T_{a−1}) / r_S, where f(T) = ([1 − T/γ]₊)^κ penalises hot summers,
   and keeps them with probability C_pers: q_a = q̃_a (1 − C_pers q_{a−1}) + C_pers q_{a−1}.
3. **Observation** (binomial). The number of trees with symptoms among the 30 × visits
   observed in quadrat i in year a is Binomial(30 × visits, q_a^i).

The first two years are imputed from smoothed prevalence maps. The ten parameters are
described in [`AshDiebackParameters`](@ref).

## Correspondence with PEOC

| PEOC | AshDieback.jl |
|:-----|:--------------|
| `parameters_of_the_model.py` | [`ModelSettings`](@ref), [`load_peoc_data`](@ref) keywords |
| `environmental_data`, `observations`, `initialisation_rachis` | [`load_peoc_data`](@ref), [`PEOCData`](@ref) |
| `reaction_diffusion.reaction_di` | [`solve`](@ref CommonSolve.solve) on an [`AshDiebackProblem`](@ref) |
| `diffusion_solver = 'exact'` / `'fast'` | [`DenseCrankNicolson`](@ref) / [`SparseCrankNicolson`](@ref) |
| `likelihood.log_likelihood` | [`peoc_loglikelihood`](@ref), [`loglikelihood`](@ref) |
| prior in `1_or_2_parameters…py` | [`AshDiebackPosterior`](@ref), [`peoc_prior_ranges`](@ref) |
| action 1 (Metropolis–Hastings) | [`ComponentwiseGammaMH`](@ref) with `AbstractMCMC.sample` |
| action 2 (AMIS) | [`amis`](@ref) |

The ten-vector order, the year offsets and PEOC's corner cases are kept, so parameter
vectors and saved chains can be exchanged with PEOC. Random draws are Julia's, so chains
agree with PEOC's in distribution, not draw by draw.
