# Analisis-de-Datos-II
MIA FIUBA - Analisis de Datos II

Author: Braian Desía (b.desia@hotmail.com)

## About this project

This repository contains different activities solved during the course.

## Project structure

The project was structured as follows.

```
├── data                                # Data. Here you can find the csv used for activities 1 and 2 (ds_salaries.csv) and 4 (census_income.csv).
│   └── fashion-mnist                   # Fashion MNIST files for activity 3 (downloaded automatically by the notebook, not tracked).
├── ACTIVIDAD 1.pdf                     # Assignment statement for activity 1.
├── ACTIVIDAD 2.pdf                     # Assignment statement for activity 2.
├── ACTIVIDAD 3.pdf                     # Assignment statement for activity 3.
├── ACTIVIDAD 4.pdf                     # Assignment statement for activity 4.
├── Actividad_1.ipynb                   # Jupyter notebook for activity 1 (executed, with outputs).
├── Actividad_2.ipynb                   # Jupyter notebook for activity 2 (executed, with outputs).
├── Actividad_3.ipynb                   # Jupyter notebook for activity 3 (executed, with outputs).
└── Actividad_4.ipynb                   # Jupyter notebook for activity 4 (executed, with outputs).
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

**Activity #3: Dimensionality reduction and clustering**

Unsupervised analysis of the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset (70,000 28x28 grayscale images of Zalando clothing items, 10 balanced classes). Each image is represented as a vector of 784 pixel intensities rescaled to [0, 1]; the work is done on a stratified sample of 10,000 images. Labels are only used for interpretation, never for fitting or choosing hyperparameters.

*Main features*

- Exploratory analysis: examples and mean image per class, intensity distribution, per-pixel variability, and cosine similarity between class prototypes (Pullover/Coat/Shirt reach 0.98-0.99).
- Dimensionality reduction with PCA: 83 components (out of 784) retain 90% of the variance; components and reconstructions are plotted as images for interpretation.
- Clustering with K-means and agglomerative (Ward) clustering, choosing k = 8 with Silhouette and Davies-Bouldin, and comparing both with internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external (ARI, AMI) metrics. K-means is selected.
- Interpretation of each cluster through its centroid reconstructed in pixel space, its class composition and its most representative images: clusters group by silhouette and tone rather than by commercial category.
- 2D visualization with UMAP on the original pixels: density map, small-multiple panels by class and by cluster, and detection of the most isolated points.

*Notebook:* [Actividad_3.ipynb](Actividad_3.ipynb) (requires `umap-learn`)

**Activity #4: Feature engineering, data leakage, bias audit and documentation**

Linear regression to predict annual income (`income`) on a census dataset of 10,000 people (age, hours per week, education, employer type, marital status, sex, race and whether they were born in the US), with a bias audit by sensitive variables and a Model Card.

*Main features*

- Target analysis: strong right skew (4.45). Log overcorrects (-1.01), Box-Cox (λ ≈ 0.21) leaves it almost symmetric (0.06) and is selected.
- Leakage-free 5-fold cross validation: imputation, splines on the numeric features, one-hot encoding and the Box-Cox transformation (`TransformedTargetRegressor`) are all refit inside each fold.
- MAE comparison: Box-Cox target lowers MAE from ~30.8k to ~26.7k USD and removes negative predictions, at the cost of underestimating the mean (retransformation bias).
- Bias audit by `sex` and `race` with out-of-fold predictions: large gaps in absolute MAE but similar relative MAE (0.46-0.50); a counterfactual test shows the model reproduces the income gaps in the data (+27% for men, +17% for White vs Black people).
- Model Card with purpose, inputs/outputs, global and per-subgroup metrics, fairness considerations and limitations.

*Notebook:* [Actividad_4.ipynb](Actividad_4.ipynb)
