# 🧬 Breast Cancer Biomarker Classifier

**Author:** Bhavya Sevak — M.S. Biomedical Informatics, Arizona State University  
**Dataset:** Wisconsin Breast Cancer Dataset (UCI / sklearn)  
**Status:** Complete

---

## Overview

This project builds a machine learning pipeline to classify breast cancer tumors as **malignant or benign** based on 30 cell nucleus features extracted from digitized fine needle aspirate (FNA) biopsy images.

The project demonstrates a full biomedical data science workflow — from exploratory analysis to model evaluation and biomarker identification — directly relevant to clinical decision support and cancer research.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~97% | ~0.997 |
| Random Forest | ~96% | ~0.995 |
| SVM | ~98% | ~0.998 |

**Top identified biomarkers:** worst radius, worst perimeter, mean concave points

---

## Project Structure

```
breast-cancer-classifier/
│
├── breast_cancer_classifier.ipynb   # Main analysis notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── plots/                           # Auto-generated visualizations
    ├── class_distribution.png
    ├── feature_boxplots.png
    ├── correlation_heatmap.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── feature_importance.png
    └── lr_coefficients.png
```

---

## Notebook Contents

1. **Data Loading & Exploration** — dataset shape, class balance, summary stats
2. **EDA** — boxplots, correlation heatmap, feature distributions
3. **Preprocessing** — StandardScaler, stratified train/test split
4. **Model Training** — Logistic Regression, Random Forest, SVM via sklearn Pipelines
5. **Evaluation** — confusion matrices, ROC curves, 5-fold cross-validation
6. **Biomarker Identification** — feature importances + logistic regression coefficients
7. **Conclusions** — clinical relevance and future directions

---

## How to Run

### Option 1 — Local
```bash
git clone https://github.com/bsevakstark/breast-cancer-classifier.git
cd breast-cancer-classifier
pip install -r requirements.txt
jupyter notebook breast_cancer_classifier.ipynb
```

### Option 2 — Google Colab (no install needed)
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsevakstark/breast-cancer-classifier/blob/main/breast_cancer_classifier.ipynb)

---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
- **Built into sklearn:** `from sklearn.datasets import load_breast_cancer`

---

## Skills Demonstrated

- Python (pandas, numpy, matplotlib, seaborn)
- Machine learning (scikit-learn Pipelines)
- Biomedical data analysis
- Cross-validation & model evaluation
- Feature importance & biomarker discovery
- Data visualization

---

## Connection to Research

This project mirrors the analytical workflow used in my mGWAS research on gallstone disease at ASU — feature selection, statistical modeling, and biomarker identification from biological datasets.

---

*Part of my biomedical informatics portfolio: [bsevakstark.github.io](https://bsevakstark.github.io)*
