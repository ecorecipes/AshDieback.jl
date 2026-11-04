# Data

These files are the input data of PEOC 1.0 (C. Fritsch, INRIA), copied unchanged from its
release, <https://github.com/coraliefritsch/peoc> at commit `f966a7d`, where they are
distributed under the CeCILL v2.1 licence (see `../LICENSE`). They are the data of Fritsch,
Grosdidier, Gégout-Petit and Marçais, *Mechanistic-statistical model for the expansion of
ash dieback* (<https://hal.science/hal-04690647>).

| file | content |
|:-----|:--------|
| `climate_data` | yearly climate per 8 km point (x, y in km, extended Lambert II), 2006–2022: June rainfall `RR6` and temperature indices `NjS24`…`NjS35` (days in July–August with a maximum over 24…35 °C) |
| `density.txt` | smoothed ash basal area (m² ha⁻¹) per point (x, y in m); old-Mac CR line endings |
| `rachis_initialization.txt` | smoothed prevalence maps `Poids07_juin21`, `Poids08_juin21` used to impute the first two years (x stored 1000 times too large, y in km) |
| `data_chalara.txt` | the DSF survey of ash dieback, one row per plot visit, 2008–2024: year (December reports moved to the next year), quadrat indices, prevalence (%) |

`load_peoc_data` reads them exactly as PEOC does; see its docstring.
