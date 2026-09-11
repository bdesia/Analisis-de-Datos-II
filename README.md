# Analisis-de-Datos-II
MIA FIUBA - Analisis de Datos II

Author: Braian Desía (b.desia@hotmail.com)

## About this project

This repository contains different activities solved during the course.

## Project structure

The project was structured as follows.

```
├── data                                # Data. Here you can find the csv used for activity 1 (ds_salaries.csv).
├── ACTIVIDAD 1.pdf                     # Assignment statement for activity 1.
└── Actividad_1.ipynb                   # Jupyter notebook for activity 1 (executed, with outputs).
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
