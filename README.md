# Machine Learning-Based Spatial Prediction of Urban RF-EMF Exposure in Tirana, Albania

This repository accompanies the manuscript of the same title and contains the
final dataset, the Python scripts validated by the doctoral supervisor, and
the publication figures and tables extracted directly from the manuscript
draft.

## Study

The study includes 171 georeferenced RF-EMF measurement points collected in:

- Mustafa Q. Atatürk Square: 85 points
- Karl Topia Square: 86 points

The dataset contains spatial variables, antenna-related variables, seven
frequency-band measurements, and total electric-field exposure.

## Repository structure

```text
data/       Corrected final dataset
scripts/    Final validated Python scripts, preserved exactly as supplied
figures/    Publication figures extracted directly from the manuscript draft
tables/     Publication tables extracted directly from the manuscript draft
docs/       Equations and publication metadata
```

## Important provenance statement

The files in `scripts/` have not been reformatted, renamed internally, or
modified. They are the exact final scripts used by the author and validated by
the doctoral supervisor.

The files in `figures/` and `tables/` were extracted directly from
`AI_Ejona_draft.docx` and renamed only to reflect their numbering in the
manuscript.

## Dataset

`data/urban_rf_emf_ai_dataset_corrected.csv`

Main RF-EMF variables:

- LTE700
- LTE800
- GSM900
- GSM1800
- LTE2100
- LTE2600
- NR3500
- E_total

## Software

The scripts use Python packages including NumPy, pandas, Matplotlib, Seaborn,
SciPy, scikit-learn, and Pillow.

## Citation

Please cite the associated article and the Zenodo DOI assigned to the GitHub
release.

## License

The code is released under the MIT License. Dataset reuse requires
attribution to the author and the associated publication.
