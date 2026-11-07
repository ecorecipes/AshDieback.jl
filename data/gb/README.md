# Great Britain data

Reduced from the sources below to OS National Grid 10 km squares, for `load_gb_inputs`,
`load_gb_future` and vignette 6. The full provenance and the reduction scripts are kept with the
project's workspace notes, not in this package.

| File | Content | Source | Licence |
|:-----|:--------|:-------|:--------|
| `ash_10km.csv` | modelled ash area (ha) and land area (km²) per square with land | Brown et al. 2024, Rothamsted Research, <https://repository.rothamsted.ac.uk/id/eprint/33927/>, doi:10.23637/mblo1mu2 | CC BY 4.0 |
| `seed_squares_10km.csv` | extract: the 734 squares whose first confirmed ash dieback find was in 2012–2015, where the model's epidemic starts | read from Forest Research's 10 km map of first finds, October 2025, <https://www.forestresearch.gov.uk/tools-and-resources/fthr/pest-and-disease-resources/ash-dieback-hymenoscyphus-fraxineus> | see below |
| `t28_haduk_10km.csv` | days over 28 °C in July and August, 2004–2025 | HadUK-Grid v1.3.2, Met Office via CEDA, <https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb> | Open Government Licence |
| `t28_chess_2026_2080.int16` | the same for model years 2026–2080, by scenario and member | CHESS-SCAPE bias-corrected, UKCEH via CEDA, <https://catalogue.ceda.ac.uk/uuid/8194b416cbee482b89e0dfbe17c5786c> | Open Government Licence |
| `kent_survey_mean.csv` | extract: the share of ash with symptoms by year, 2017–2024, as the mean over the survey's tetrads with a value that year | computed from the charts of Butfoy & Harwood, June 2024, Kent County Council, <https://www.kent.gov.uk/__data/assets/pdf_file/0008/181358/Ash-Dieback-survey-data-for-Kent.pdf> | see below |
| `kent_squares_10km.csv` | the 10 km square of each of the survey's nine tetrads, placed by hand from their place names, so approximate | as above | see below |

The reuse terms of the Forest Research map (which carries Crown copyright and Ordnance Survey
database right) and of the Kent survey were not confirmed. So this package holds only these
extracts, with attribution: which squares seed the model, rather than the map's periods of first
find, and the survey's yearly means, rather than the figures for each tetrad.

`t28_chess_2026_2080.int16` holds little-endian Int16 hundredths of a day, −1 where a square has no
value, in the order scenario (`rcp45`, `rcp85`), member (`01`, `04`, `06`, `15`), square (as in
`ash_10km.csv`), year. CHESS-SCAPE's RCP4.5 is scaled from its RCP8.5, its years are model years, and
its 360-day counts are already scaled by 62/60.
