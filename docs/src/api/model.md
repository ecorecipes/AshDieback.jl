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
CommonSolve.solve
SciMLBase.remake(::AshDiebackProblem)
AshDiebackSolution
initial_fungi
symptom_probability
```

## A changing host

```@docs
HostClass
HostDynamics
constant_host
HostProblem
HostSolution
host_years
live_host
symptom_share
```

## Interventions

```@docs
AbstractIntervention
Vaccinate
vaccinated_class
PlantStock
Fell
ReduceLitter
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
