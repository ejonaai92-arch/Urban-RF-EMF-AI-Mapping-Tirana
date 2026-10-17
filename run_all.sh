#!/usr/bin/env bash
# Reproduces all v2.0.0 results and revised figures. Run from the repository root.
set -euo pipefail
cd scripts
python 00_base_stations.py
python 01_validation.py          # Table 2, Table 5, bootstrap CIs (~5 min)
python 02_oof_hotspots.py        # OOF predictions, Table 4
python 03_interpolation_loocv.py # Table 3, variograms
python 04_learning_curves.py     # Fig. 12 data
python 05_daytype.py             # weekday/weekend analyses
python 06_sampling_geometry.py   # spacing and density
python 07_kriging_sd.py          # kriging standard deviation range
python 10_load_spectra.py        # raw Narda exports -> 171 x 531 spectra
python 11_spectral_analysis.py   # corrected levels, uplink share, ICNIRP ratios
python 12_point_metadata.py      # field sheets -> location type, session times
python 20_figures.py             # Figs. 1, 3, 9-12
python 21_figures_6_8.py         # Figs. 6-8 (single split)
