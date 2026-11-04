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
DegenerateProposalError
```

## Turing

```@docs
turing_model
```
