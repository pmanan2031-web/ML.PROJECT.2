<div align="center" id="top">

# 🏠 House Price Prediction

**Machine Learning Regression Project**

<p>
<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white">
</p>

<a href="./project2.ipynb">📓 NOTEBOOK</a> •
<a href="./house_price.csv">📊 DATASET</a> •
<a href="#workflow">⚙️ WORKFLOW</a> •
<a href="#results">📈 RESULTS</a>

<br><br>

<img src="./assets/model_comparison.gif" alt="Model Comparison" width="720">

</div>

---

## 📌 About

A complete **House Price Prediction** project using real-estate data.  
The project covers preprocessing, regularization, cross-validation, tree models, SVR, and final model evaluation.

**Target:** `houseprice`  
**Dataset:** `house_price.csv` — 3,800 rows, 12 columns. fileciteturn0file1

---

## 🎯 What I Did

- 🧹 Data preparation & feature selection
- ✂️ Train-Test Split
- 📏 Feature Scaling
- 📉 Ridge & Lasso Regression
- 🔄 K-Fold, Stratified K-Fold, LOOCV & Time Series Split
- 🌳 Decision Tree Regression
- 🌲 Random Forest Regression
- 📐 SVR — Linear, Polynomial & RBF
- 🎛️ Hyperparameter tuning
- 📊 MSE, MAE, RMSE & R² evaluation

These are the main modelling components used in the project. fileciteturn0file0

---

## ⚙️ Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Train / Test Split
   ↓
Scaling
   ↓
Ridge + Lasso
   ↓
Cross-Validation
   ↓
Decision Tree + Random Forest
   ↓
SVR
   ↓
Model Comparison
   ↓
🏆 Best Model
```

---

## 🤖 Models

| Category | Models |
|---|---|
| 📉 Regularized Linear | Ridge, Lasso |
| 🌳 Tree-Based | Decision Tree, Random Forest |
| 📐 SVR | Linear, Polynomial, RBF |

The notebook also tunes Ridge/Lasso regularization and SVR parameters. fileciteturn0file3

---

## 📈 Results

| Model | Test R² |
|---|---:|
| Ridge | 0.9187 |
| Lasso | 0.9187 |
| Decision Tree | 0.8835 |
| **Random Forest** | **0.9269** |
| SVR Linear | 0.0983 |
| SVR Polynomial | 0.0035 |
| SVR RBF | -0.0004 |

### 🏆 Best Model: Random Forest

**Test R² = 0.9269**  
**Test MSE ≈ 5.89 × 10¹²** fileciteturn0file4

---

## 🛠️ Tech Stack

`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Scikit-Learn` • `Jupyter Notebook`

---

## 📁 Structure

```text
House-Price-Prediction/
├── project2.ipynb
├── house_price.csv
├── README.md
└── assets/
    └── model_comparison.gif
```

---

## ▶️ Run Project

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook
```

Open `project2.ipynb` and run **Part B → C → D → E → F → G**. fileciteturn0file0

---

<div align="center">

### ⭐ If you like this project, give it a star!

<a href="#top">⬆️ Back to Top</a>

</div>
