# Random Forest Prediction of Urban RF-EMF Exposure from Multiband Field Measurements in Tirana

Code, data and figures for the manuscript *Random Forest Prediction of Urban Radiofrequency Electromagnetic Field Exposure from Multiband Field Measurements in Tirana* (Zeneli, Kuqi, Myzeqari, Kuqali, Askushaj, Plaku; revised version submitted to *Discover Applied Sciences*).

**Version 1.1.0** adds everything needed to reproduce the analyses introduced in the revision (cross-validation, bootstrap, spatial-only and linear baselines, kriging comparison, hotspot agreement, learning curves, weekday/weekend sensitivity, uplink and noise-corrected levels) together with the revised figures. Version 1.0.0 (original submission) is preserved unchanged in `scripts/v1_original/`, `figures/v1_original/` and `tables/v1_original/`.

## Study

171 georeferenced spot measurements (Narda SRM-3006, three-axis E-field antenna 420 MHz–6 GHz, 700 MHz–6 GHz sweep, RBW 20 MHz, 10 MHz step, 531 bins, 6-min averages) in two squares of Tirana, Albania, 19 April–5 May 2026:

| Square in manuscript | `Zone` in dataset | Points | Base stations |
|---|---|---|---|
| Mustafa Kemal Atatürk Square (MKA) | `21_Dhjetori` | 85 | BS-M1, BS-M2, BS-M3 |
| Karl Topia Square (KT) | `Zogu_i_Zi` | 86 | BS-K1, BS-K2 |

The measurement campaign itself (protocol, uncertainty, band-wise weekday–weekend and diurnal analysis, IDW maps, ICNIRP compliance) is reported in Kuqi et al., *Applied Sciences* 2026, 16, 8676. This repository covers the machine-learning analysis of the same measurements.

## Repository structure

```text
data/
  urban_rf_emf_ai_dataset.csv     171 points: coordinates, antenna predictors, 7 band fields, E_total (unchanged since v1.0.0)
  base_stations.csv               the 5 surveyed base stations (WGS84, antenna elevation a.s.l.)     [new]
  point_metadata.csv              day type, UTM 34N coordinates, location type, session times      [new]
  spectra_Eact_531bins.csv        full 700 MHz–6 GHz Eact spectrum of every point (V/m)            [new]
  spectra_source_files.csv        point -> original Narda file, index, instrument time and settings [new]
  point_spectral_summary.csv      noise/RBW-corrected field, FDD uplink share, NR3500 share, ICNIRP ratios [new]
  raw_spectra/                    original Narda SRM-3006 exports: <Zone>/Dt-<day>/P<nn>/ (ACT/MAX/MIN spectra,
                                  header, screenshots); Export_from_Narda/ = same files as exported   [new]
  field_sheets/                   field sheets (Flete_Matje_*.xlsx) and base-station survey          [new]
scripts/
  common.py                       paths, constants, loaders
  00_base_stations.py             writes data/base_stations.csv
  01_validation.py                Table 2, Table 5, bootstrap CIs
  02_oof_hotspots.py              10-fold out-of-fold predictions, Table 4
  03_interpolation_loocv.py       Table 3, variogram parameters
  04_learning_curves.py           Fig. 12 data
  05_daytype.py                   weekday/weekend tests and model sensitivity
  06_sampling_geometry.py         point spacing, extents, densities
  07_kriging_sd.py                range of the kriging standard deviation (Section 3.4)
  10_load_spectra.py              raw spectra -> data/spectra_Eact_531bins.csv
  11_spectral_analysis.py         corrected levels, uplink share, ICNIRP ratios
  12_point_metadata.py            field sheets -> data/point_metadata.csv
  20_figures.py                   revised Figs. 1, 3, 9–12
  21_figures_6_8.py               Figs. 6–8 (single 75/25 split, Table 2 importances)
  v1_original/                    the seven v1.0.0 scripts (Figs. 4–8, original Tables)
results/                          CSV/TXT outputs of the scripts above
figures/                          figures of the revised manuscript (Figs. 1, 3–12; Fig. 2 is a photograph)
tables/v1_original/               tables of the original submission
run_all.sh                        runs the full pipeline
```

## Where each result of the revised manuscript comes from

| Manuscript item | Script | Output |
|---|---|---|
| Section 2.1 spacing, density, schedule | `06_sampling_geometry.py`, `05_daytype.py` | `results/sampling_geometry.csv`, `results/daytype_sensitivity.txt` |
| Fig. 1 (points, location type, base stations) | `12_point_metadata.py`, `20_figures.py` | `figures/Figure_01_*.png` |
| Section 2.2 / 3.1 corrected levels, uplink share, ICNIRP | `10_load_spectra.py`, `11_spectral_analysis.py` | `data/point_spectral_summary.csv`, `results/spectral_summary.txt` |
| Table 1, Figs. 4–5 | `v1_original/` (unchanged) | `figures/Figure_04`, `Figure_05` |
| Figs. 6–8 (single split, Table 2 importances) | `01_validation.py`, `21_figures_6_8.py` | `figures/Figure_06`–`08` |
| Section 3.1 weekday vs weekend | `05_daytype.py` | `results/daytype_sensitivity.txt`, `results/daytype_paired_locations.csv` |
| Table 2 feature importance | `01_validation.py` | `results/table2_feature_importance.csv` |
| Table 3, variograms, kriging SD | `03_interpolation_loocv.py`, `07_kriging_sd.py` | `results/table3_interpolation_loocv.csv`, `results/variogram_parameters.csv`, `results/kriging_sd_range.csv` |
| Figs. 9–11, Table 4 | `02_oof_hotspots.py`, `20_figures.py` | `results/oof_predictions_hotspots.csv`, `results/table4_hotspot_agreement.csv` |
| Table 5, bootstrap CIs, spatial-only importances | `01_validation.py` | `results/table5_validation.csv`, `results/cv_fold_metrics.csv`, `results/bootstrap_ci_single_split.csv`, `results/spatial_only_importance.csv` |
| Fig. 12 learning curves | `04_learning_curves.py`, `20_figures.py` | `results/learning_curves_*.csv` |

## Data dictionary (`data/urban_rf_emf_ai_dataset.csv`)

| Column | Description |
|---|---|
| `Point_ID` | point identifier (`DH_` = MKA, `ZZ_` = KT) |
| `Zone` | `21_Dhjetori` = MKA, `Zogu_i_Zi` = KT |
| `Date`, `Time` | local date (**MM/DD/YYYY**) and time of the measurement |
| `Latitude`, `Longitude`, `Altitude_m` | instrument GPS (DGPS), WGS84 (EPSG:4326) |
| `Nearest_Antenna_ID` | `21_Djetori` (MKA stations) or `Zogu_Zi` (KT stations) |
| `Distance_to_nearest_antenna_m` | haversine distance to the nearest of the five stations in `base_stations.csv` |
| `Nearest_Antenna_Height_m` | elevation above sea level of that station's antenna, as recorded in the survey (feet × 0.3048); called "antenna height" in the manuscript |
| `LTE700` … `NR3500` | RSS of the Eact bins within each band's 3GPP allocation (V/m) |
| `E_total` | RSS of all 531 bins, 700 MHz–6 GHz (V/m), Eq. (1); includes the accumulated noise floor |

Maps and distances in the manuscript use UTM zone 34N (EPSG:32634); `point_metadata.csv` gives these coordinates.

## Reproducing the results

```bash
pip install -r requirements.txt
bash run_all.sh
```

Random seeds are fixed in every script. Notes:

* **Column order.** Random Forest draws a random permutation of the features at each split, so results depend slightly on column order. The single 75/25 split uses the column order of the v1.0.0 script (spatial predictors first), which reproduces the headline metrics R² = 0.806, MAE = 0.231 V/m, RMSE = 0.321 V/m. Cross-validation, Table 2 and all other analyses use band-first order. With band-first order the single split gives R² = 0.805; the per-area bootstrap R² (v1 order) is 0.788 (KT) and 0.818 (MKA), as reported in the revised manuscript. These differences are well within the reported confidence intervals.
* **Day-type sensitivity** (`05_daytype.py`) reports the pooled out-of-fold R² over ten repetitions of 10-fold cross-validation (one R² per repetition from all 171 out-of-fold predictions, then averaged), which differs from the fold-mean R² of Table 5.
* **Cross-validation SDs** in Table 5 are population SDs over the 50 or 100 fold-level evaluations.
* **Kriging.** A 5 cm jitter (fixed seed) separates re-measured points with identical coordinates; surfaces are clipped to the convex hull of the points and used for visualization only.
* The scripts in `scripts/v1_original/` are kept exactly as released in v1.0.0. The file name `urban_rf_emf_ai_dataset_corrected.csv` mentioned in the v1.0.0 documentation refers to the same data as `urban_rf_emf_ai_dataset.csv` (identical content).
* **Raw spectra.** Each dataset row is linked to its spectrum by the instrument date and time (`data/spectra_source_files.csv`); the RSS of every linked spectrum equals `E_total` to machine precision. For `ZZ_P084` the dataset time (13:13:00) differs from the instrument time (13:17:47); it is linked by its identical RSS. Four further spectra in the raw folders (KT, 19 April P03, P05, P06 and 20 April P12) are repeated recordings that were not used.
* **Figs. 6–8.** The PNG files in `figures/` are the figures of the revised manuscript. `21_figures_6_8.py` regenerates them from the same split and importances (identical points, metrics and bar values); in Fig. 7 one residual lies on a histogram bin edge and can fall into the adjacent bin depending on library versions.
* **Session times.** The schedule in Section 2.1 (08:26–18:35) refers to the 6-min measurement sessions recorded in the field sheets (`Session_start`, `Session_end` in `point_metadata.csv`); `Time` in the dataset is the instrument save time at the end of each averaging window.
* **Uplink share** is computed for the five FDD bands of Section 3.1 (LTE800, GSM900, GSM1800, LTE2100, LTE2600).

## Citation

Please cite the associated article and this Zenodo record (see `CITATION.cff`).

## License

Code: MIT License. Data: reuse with attribution to the authors and the associated publications.
