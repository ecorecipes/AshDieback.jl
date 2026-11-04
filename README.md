# AshDieback.jl

The mechanistic-statistical model of the spread of ash dieback (*Hymenoscyphus fraxineus*)
across France of Fritsch, Grosdidier, Gégout-Petit and Marçais
([HAL 04690647](https://hal.science/hal-04690647)), with its data and its Bayesian
inference. It is a Julia port of [PEOC](https://github.com/coraliefritsch/peoc), the
authors' Python code, and reproduces its log-likelihoods to a relative 1e-14; a likelihood
takes under 2 s, against about a minute for PEOC 1.0.

## Features

- **The data**, bundled and read exactly as PEOC reads them: climate, ash density, the
  survey of 2008–2023 and the prevalence maps used for initialisation.
- **SciML-style problem and solve**: `AshDiebackProblem`, `solve`, `remake`; the spore
  diffusion is a Crank–Nicolson scheme solved through LinearSolve.jl with one sparse Cholesky
  factorisation per likelihood, checked elementwise against PEOC's dense scheme, tails
  included.
- **Likelihood and posterior**: `loglikelihood`, PEOC's exact `peoc_loglikelihood`, and the
  posterior under uniform priors as a LogDensityProblems.jl density.
- **PEOC's samplers**: the component-wise gamma-proposal Metropolis–Hastings as an
  AbstractMCMC.jl sampler (with MCMCChains output) and adaptive multiple importance sampling
  (AMIS), threaded.
- **Turing.jl** model of the posterior (package extension), and Plots.jl recipes for maps.

## Quick start

```julia
using AshDieback

data = load_peoc_data()                     # the bundled PEOC 1.0 data, T28 index
θ = AshDiebackParameters(; r_symp = 1000.0, r_allee = 0.0, beta_0 = 30.0, beta_1 = 0.0,
                         D = 20.0, S_sat = 90.0, gamma = 22.0, C_pers = 1.0,
                         C_init = 0.0085, kappa = 0.1)
prob = AshDiebackProblem(data, θ)
sol = solve(prob)                           # colonised rachis and symptom probabilities
loglikelihood(sol)                          # -25151.766135874...

# the posterior of the paper's reduced model, sampled with PEOC's Metropolis-Hastings
using AbstractMCMC, MCMCChains
post = AshDiebackPosterior(prob, reduced_model_prior_ranges())
chain = sample(post, ComponentwiseGammaMH(post), 1_000;
               initial_params = θ[post.free], chain_type = Chains)
```

## Vignettes

| # | Vignette | Description |
|---|----------|-------------|
| 1 | [The ash dieback model](vignettes/01_introduction/01_introduction.md) | Data, simulation, maps and the likelihood, checked against PEOC |
| 2 | [Inference](vignettes/02_inference/02_inference.md) | The posterior, Metropolis–Hastings, AMIS and Turing |

## Installation

Not registered. From a clone: `julia --project -e 'using Pkg; Pkg.instantiate()'`.

## Licence and provenance

PEOC and its data are distributed under the CeCILL v2.1 licence by Coralie Fritsch (INRIA);
this port is a derivative work under the same licence (`LICENSE`). The data in `data/` are
PEOC 1.0's, unchanged.

## Related

- [PEOC](https://github.com/coraliefritsch/peoc), the original Python 2 code.
- [EcologicalCellularAutomata.jl](https://github.com/ecorecipes/EcologicalCellularAutomata.jl):
  published ecological cellular automata, including plant diseases.
