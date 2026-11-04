# Model

```@meta
CurrentModule = AshDieback
```

## Parameters and settings

```@docs
AshDiebackParameters
parameter_names
ModelSettings
simulated_years
survey_years
temperature_effect
```

## Problem and solution

```@docs
AshDiebackProblem
CommonSolve.solve(::AshDiebackProblem, ::AbstractDiffusionAlgorithm)
SciMLBase.remake(::AshDiebackProblem)
AshDiebackSolution
initial_fungi
symptom_probability
```

## Diffusion

```@docs
AbstractDiffusionAlgorithm
SparseCrankNicolson
DenseCrankNicolson
neumann_laplacian
```

## Likelihood

```@docs
loglikelihood
peoc_loglikelihood
BinomialTerms
```
