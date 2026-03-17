# 🧬 Breast Cancer Biomarker Classifier

**Author:** Bhavya Sevak — M.S. Biomedical Informatics, Arizona State University  
**Dataset:** Wisconsin Breast Cancer Dataset (UCI / sklearn)  
**Live Demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Overview

A full machine learning pipeline to classify breast cancer tumors as **malignant or benign** using 30 cell nucleus features from digitized FNA biopsy images — plus an interactive Streamlit dashboard with SHAP explainability.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~97% | ~0.997 |
| Random Forest | ~96% | ~0.995 |
| SVM | ~98% | ~0.998 |

**Top biomarkers identified:** worst radius, worst perimeter, mean concave points

---

## Features

- 📊 **Interactive EDA** — class distribution, boxplots, correlation heatmap
- 🤖 **3 ML Models** — Logistic Regression, Random Forest, SVM
- 📈 **Evaluation** — ROC curves, confusion matrices, classification reports
- 🔍 **SHAP Explainability** — summary plots, waterfall charts, per-sample explanations
- 🎛️ **Live Predictor** — adjust sliders, get real-time predictions with confidence scores

---

## Project Structure

```
breast-cancer-classifier/
│
├── app.py                           # Streamlit dashboard
├── breast_cancer_classifier.ipynb   # Full analysis notebook (with SHAP)
├── requirements.txt                 # Dependencies
├── README.md
└── plots/                           # Auto-generated visualizations
```
## Skills Demonstrated

`Python` `scikit-learn` `SHAP` `Streamlit` `pandas` `matplotlib` `seaborn` `Jupyter` `Machine Learning` `Explainable AI` `Biomarker Discovery`

---

*Part of my portfolio: [bsevakstark.github.io](https://bsevakstark.github.io)*
