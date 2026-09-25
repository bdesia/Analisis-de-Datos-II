# Analisis-de-Datos-II
MIA FIUBA - Analisis de Datos II

Author: Braian Desía (b.desia@hotmail.com)

## About this project

This repository contains different activities solved during the course.

## Project structure

The project was structured as follows.

```
├── data                                # Data. Here you can find the csv used for activities 1 and 2 (ds_salaries.csv).
├── ACTIVIDAD 1.pdf                     # Assignment statement for activity 1.
├── ACTIVIDAD 2.pdf                     # Assignment statement for activity 2.
├── Actividad_1.ipynb                   # Jupyter notebook for activity 1 (executed, with outputs).
└── Actividad_2.ipynb                   # Jupyter notebook for activity 2 (executed, with outputs).
```

## Activities

**Activity #1: Model interpretability**

Audit of a Random Forest regressor trained on the [Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023) dataset, using global (PFI) and local (SHAP) interpretability techniques to diagnose the model's behavior.

*Main features*

- Global diagnosis with Permutation Feature Importance (MAE-based, measured in dollars).
- Local diagnosis with SHAP (`TreeExplainer`): waterfall plots for the highest and lowest predictions, plus a beeswarm summary.
- Model audit: detection of data leakage (`salary` is the target in local currency), redundant geographic features, and duplicated rows.
- Retraining without the leaking feature and comparison of performance (R² 0.930 → 0.359), feature importances and behavior.

*Notebook:* [Actividad_1.ipynb](Actividad_1.ipynb)

**Activity #2: Data quality, drift and anomalies**

Analysis of data quality, distribution drift and anomalies on the same [Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023) dataset, focusing on `experience_level`, `salary_in_usd` and `company_location`. The data is split into a reference period (2020-2021, 306 rows) and a current period (2022-2023, 3,449 rows).

*Main features*

- Quality check with Great Expectations: completeness, validity, data type and uniqueness.
- Drift evaluation with the Population Stability Index (Evidently `ValueDrift(method="psi")`). Includes a visual comparison of the distributions per period.
- Anomaly detection with Isolation Forest and Local Outlier Factor on the distinct combinations of the three variables; 12 of 29 flagged instances are shared by both methods.
- Analysis of one outlier flagged by both methods (`EX` / 15,000 USD / `CA`), showing it is a multivariate anomaly rather than a univariate one.

*Notebook:* [Actividad_2.ipynb](Actividad_2.ipynb)
