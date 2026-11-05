# AshDieback.jl

The mechanistic-statistical model of ash dieback spread across France (Fritsch, Grosdidier,
Gégout-Petit & Marçais, <https://hal.science/hal-04690647>), its data and its inference: a
Julia port of PEOC 1.0 (<https://github.com/coraliefritsch/peoc>) through its Python 3 port
`../peoc_py3`. The workspace notes on the model, the port and its verification are the OKF
wiki in `../../wiki` (start at `index.md`; `port/reimplementation-guide.md` is the spec this
package was built from).

## Invariants that must not be broken

- The data layer reproduces PEOC's arrays bitwise (`test/test_data.jl`): coordinate rounding
  ties to even, Corsica dropped at x ≥ 1 047 000 m, the −1 fill of the climate arrays,
  the string filter `!= "0"` on the rachis file, repeat survey visits merged in file order.
- Log-likelihoods match the PEOC 1.0 oracle (`test/reference/peoc_reference.toml`, from
  Python 2.7) to a relative 1e-10 (measured: ≤ 8e-15), and the fungus field matches
  elementwise *in relative terms*, tails included (measured: ≤ 1.7e-12). The likelihood reads the field's tails (down to 1e-134) through
  log(q): never solve the diffusion with an FFT/DCT or an adaptive ODE solver for the
  likelihood; keep a Crank–Nicolson step whose factors are M-matrices (Cholesky, or LU with
  diagonal pivots).
- Year offsets: `fungi[:, :, l]` is year `2007 + l - 1`; survey year a uses F and T of
  year a − 1 (`wiki/model/notation.md`).
- `peoc_loglikelihood` keeps PEOC's quirks (flag reset by q = 1 quadrats, partial sum,
  NaN); `loglikelihood` is the corrected one (−Inf for a zero likelihood). Do not merge them.
- Arrays are `(nx, ny)` = `(68, 67)`, indexed `[x, y]`, flattened column-major; PEOC's are
  `(year, x, y)` and row-major. `generate_reference.py` converts indices.

## Layout

- `src/AshDieback.jl`: module docstring, exports, includes.
- `src/errors.jl`: `AshDiebackError` and `DataFormatError`; `DegenerateProposalError` is in
  `src/amis.jl`.
- `src/grid.jl`, `src/data.jl`: the quadrat grid and the PEOC file readers.
- `src/parameters.jl`, `src/problem.jl`: parameters, settings, `AshDiebackProblem`, `solve`.
- `src/diffusion.jl`: Laplacian and the Crank–Nicolson algorithms (`SparseCrankNicolson`
  through LinearSolve.jl; `DenseCrankNicolson`, PEOC's reference, for tests).
- `src/likelihood.jl`, `src/posterior.jl`: likelihoods; the LogDensityProblems posterior.
- `src/mcmc.jl`, `src/amis.jl`: PEOC's samplers (AbstractMCMC sampler; AMIS).
- `src/validation.jl`: the RMSE of Table 4 and the likelihood on a regular grid.
- `ext/`: Turing (`turing_model`) and MCMCChains (`bundle_samples`) extensions.
- `data/`: PEOC 1.0's input data, unchanged (CeCILL v2.1, see `data/README.md`).
- `test/reference/`: the oracle and the script that generates it from a PEOC dump.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite (testsets ~80 s once loaded)
cd vignettes && quarto render NN_name/NN_name.qmd                # HTML, PDF (lualatex) and GitHub Markdown
cd vignettes && quarto render NN_name/NN_name.qmd --to gfm       # then again: see below
julia scripts/sync_vignettes.jl [--check]                        # copy vignettes into docs/src/tutorials
julia --project=docs docs/make.jl                                # build docs locally
```

Rendering all formats at once deletes `NN_name_files/`, which the Markdown output (and so
the docs) needs. Re-render `--to gfm` afterwards to restore it. Quarto's Julia engine uses
a single QuartoNotebookRunner server, shared by every process on the machine. Never close
or kill it (`quarto call engine julia status` is read-only). Quarto waits only about 10 s
for a new server to start, so on a loaded machine the first render fails with "No
transport file was found"; render again once `status` shows the server running.
MCMCChains' default `plot(chain)` needs StatsPlots.jl, which the vignettes do not load: use
`seriestype = :traceplot`.

CI (`.github/workflows/CI.yml`) runs the tests on Julia 1.11 and the latest release on Linux,
macOS and Windows, and deploys the docs to GitHub Pages from `main`. It uses the committed
`docs/src/tutorials`, so re-render and sync the vignettes before pushing changes to them.

One likelihood takes about 0.7 s on a quiet machine (6000 sparse triangular solves in CHOLMOD;
LinearSolve.jl adds no measurable overhead, and CHOLMOD's in-place `ldiv!` halves the allocation
but is no faster), against 0.9–1.3 s for the Python port's `fast` solver. With 8 threads the
throughput is one every 0.15–0.2 s. The dense reference takes about a minute.

`scripts/run_amis.jl` runs the paper's AMIS (60 iterations of 10 000 draws, about a day on 8
threads) into the git-ignored `runs/`, with a checkpoint after every iteration, and resumes from
it. `scripts/summarise_amis.jl` compares a run with the paper. Do not change `AMISResult` while a
run depends on its checkpoint: it is a `Serialization` file.

## Files not to edit by hand

- `docs/src/tutorials/` is generated by `scripts/sync_vignettes.jl`.
- `vignettes/*/*.md`, `*.html` and `*_files/` are quarto output; edit the `.qmd`.
- `test/reference/peoc_reference.toml` is generated by `generate_reference.py`.
- `test/reference/peoc_validation.toml` is generated by `generate_validation_reference.py`.

## Style

JuliaFormatter `yas`; docstrings on every exported name, which `test/test_docstrings.jl`
enforces; typed exceptions under `AshDiebackError` with the offending names in the message;
invalid arguments raise `ArgumentError`; seeded RNGs only in tests; every optimised code
path is tested against a slower reference; no emojis in code or docs.

## Licence

PEOC is CeCILL v2.1, so this port, a derivative work, is too (`LICENSE`).
