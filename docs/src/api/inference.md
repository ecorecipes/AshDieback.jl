# Inference

```@meta
CurrentModule = AshDieback
```

## Posterior

```@docs
AshDiebackPosterior
peoc_prior_ranges
reduced_model_prior_ranges
free_parameters
full_parameters
posterior_problem
parameter_bounds
logprior
```

## Metropolis–Hastings

```@docs
ComponentwiseGammaMH
peoc_mh_shapes
GammaMHTransition
GammaMHState
```

## AMIS

```@docs
amis
AMISResult
amis_draws
effective_sample_size
best_sample
weighted_quantile
DegenerateProposalError
```

## Validation

```@docs
rmse
RMSETable
```

## Likelihood on a regular grid

```@docs
likelihood_grid
LikelihoodGrid
peoc_grid_axes
profile_likelihood
best_parameters
```

## Turing

```@docs
turing_model
```
