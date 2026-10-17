# Release v2.0.0

Accompanies the revised manuscript *Random Forest Prediction of Urban Radiofrequency Electromagnetic Field Exposure from Multiband Field Measurements in Tirana* (revision following peer review, Discover Applied Sciences).

## Added
- `data/base_stations.csv`: the five surveyed base stations (BS-K1, BS-K2, BS-M1–M3); reproduces the distance and height predictors exactly.
- `data/point_metadata.csv`: day type, UTM 34N coordinates, location type (Cluster/Contour/Urban) and session times.
- `data/raw_spectra/` (original Narda SRM-3006 exports, 171 measurement points plus 4 repeated recordings), `data/field_sheets/` (field sheets, base-station survey), `data/spectra_Eact_531bins.csv` and `data/spectra_source_files.csv`.
- `data/point_spectral_summary.csv`: noise/RBW-corrected total field, FDD uplink share, NR3500 share, ICNIRP 2020 exposure ratios.
- Scripts `00`–`20` for all analyses added in the revision: repeated 5-/10-fold CV, bootstrap CIs, spatial-only RF, MLR baseline, 10-fold out-of-fold predictions and P85 hotspot agreement, LOOCV of interpolators (Delaunay, IDW, RBF, ordinary kriging), variograms, learning curves, weekday/weekend tests and model sensitivity, sampling geometry, spectral analyses.
- `results/`: outputs behind Tables 2–5, Section 2.1, Section 3.1, Section 3.4 (kriging standard deviation, `07_kriging_sd.py`) and Fig. 12.
- Revised figures: Fig. 1 (new, with base stations and location types), Fig. 3 (redrawn), Figs. 6–8 (redrawn, single split and Table 2 importances; script `21_figures_6_8.py`), Figs. 9–11 (ordinary kriging, UTM 34N), Fig. 12 (new).
- `run_all.sh`, pinned `requirements.txt`, data dictionary in `README.md`.

## Unchanged
- `data/urban_rf_emf_ai_dataset.csv` (identical to v1.0.0).
- v1.0.0 scripts, figures and tables, moved to `*/v1_original/`.
