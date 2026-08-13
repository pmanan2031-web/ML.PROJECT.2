# 🏠 House Price Prediction & Regression Model Comparison

<div align="center">

# 📊 Machine Learning Regression Project

### Predicting House Prices using Multiple Machine Learning Models

<br>

<img src="assets/house.gif" alt="House Price Prediction GIF" width="500">

<br><br>

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

</div>

---

## 📌 Project Overview

This project focuses on **predicting house prices using Machine Learning Regression techniques**.

Different regression models are trained and evaluated to understand which model performs best on the house price dataset.

The project includes:

* 📂 Data Loading
* 🧹 Data Preparation
* 🎯 Feature and Target Selection
* ✂️ Train-Test Split
* ⚖️ Feature Scaling
* 📈 Ridge Regression
* 📉 Lasso Regression
* 🔄 Cross Validation
* 🌳 Decision Tree Regression
* 🌲 Random Forest Regression
* 📊 Support Vector Regression
* 🧮 Model Performance Evaluation
* 🏆 Best Model Selection
* 🔍 Regularization Analysis

---

# 🎬 Project Workflow

<div align="center">

<img src="assets/ml-workflow.gif" alt="Machine Learning Workflow" width="700">

</div>

```text
                HOUSE PRICE DATASET
                         │
                         ▼
                  DATA LOADING
                         │
                         ▼
              DATA PREPROCESSING
                         │
                         ▼
           FEATURE / TARGET SELECTION
                         │
                         ▼
               TRAIN - TEST SPLIT
                         │
                         ▼
                FEATURE SCALING
                         │
                         ▼
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
 Ridge Regression   Lasso Regression   Cross Validation
      │                  │                  │
      └──────────────────┼──────────────────┘
                         ▼
                DECISION TREE
                         │
                         ▼
                RANDOM FOREST
                         │
                         ▼
               SUPPORT VECTOR REGRESSION
                         │
                         ▼
               MODEL EVALUATION
                         │
                         ▼
                BEST MODEL SELECTION 🏆
```

---

# 📁 Project Structure

```text
House-Price-Prediction/
│
├── project2(3).ipynb
│
├── house_price.csv
│
├── README.md
│
└── assets/
    │
    ├── house.gif
    ├── ml-workflow.gif
    ├── ridge-lasso.gif
    ├── cross-validation.gif
    ├── decision-tree.gif
    ├── random-forest.gif
    └── model-comparison.gif
```

---

# 🛠️ Technologies & Tools Used

<div align="center">

| 🧰 Tool / Library   | 🎯 Purpose                         |
| ------------------- | ---------------------------------- |
| 🐍 Python           | Main Programming Language          |
| 🐼 Pandas           | Data Loading and Data Manipulation |
| 🔢 NumPy            | Numerical Calculations             |
| 📊 Matplotlib       | Data Visualization                 |
| 🤖 Scikit-learn     | Machine Learning Models            |
| 📁 CSV Dataset      | House Price Data Storage           |
| 📓 Jupyter Notebook | Project Development                |

</div>

---

# 📦 Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    RidgeCV,
    LassoCV
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.pipeline import make_pipeline

import matplotlib.pylab as plt
```

---

# 📂 Dataset

The project uses a **House Price Dataset** stored in CSV format.

```python
data = pd.read_csv("house_price.csv")
```

The dataset is loaded using **Pandas**.

Initial data inspection:

```python
print(data.head())
print(data.info())
```

### 📊 Dataset Processing

The project identifies:

```text
Input Features  → X
Target Variable → y
```

The target variable used for prediction is:

```text
houseprice
```

The column names are cleaned before processing:

```python
data.columns = data.columns.str.strip()
```

---

# 🟦 PART B — Data Preparation

## 1️⃣ Feature and Target Selection

Feature variables:

```python
X = data.drop("houseprice", axis=1)
```

Target variable:

```python
y = data["houseprice"]
```

### 🎯 Goal

The model learns the relationship between house-related features and the final **house price**.

---

## 2️⃣ Removing Unwanted Columns

For model training, the project removes:

```text
houseprice
sale_date
```

Code:

```python
X = data.drop(["houseprice", "sale_date"], axis=1)
```

---

## 3️⃣ Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

Code:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Why?

🟢 **Training Data** → Used to train the model

🔵 **Testing Data** → Used to check model performance on unseen data

---

## 4️⃣ Feature Scaling

The project uses:

> **StandardScaler**

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

### Scaling Formula

```text
Scaled Value = (X - Mean) / Standard Deviation
```

Scaling is especially useful for models such as:

* Ridge Regression
* Lasso Regression
* Support Vector Regression

---

# 🟩 PART C — Regularization Techniques

<div align="center">

<img src="assets/ridge-lasso.gif" alt="Ridge and Lasso GIF" width="550">

</div>

## 🔵 Ridge Regression

Ridge Regression uses **L2 Regularization**.

```python
ridge = Ridge(alpha=1)

ridge.fit(X_train_scaled, y_train)

ridge_pred = ridge.predict(X_test_scaled)
```

### Evaluation

```python
ridge_mse = mean_squared_error(y_test, ridge_pred)

ridge_r2 = r2_score(y_test, ridge_pred)
```

### Ridge Objective

```text
Loss = MSE + λ × Σ(coefficient²)
```

### Purpose

✔ Reduces overfitting
✔ Controls large coefficients
✔ Works well when features are correlated

---

## 🟠 Lasso Regression

Lasso Regression uses **L1 Regularization**.

```python
lasso = Lasso(alpha=1)

lasso.fit(X_train_scaled, y_train)

lasso_pred = lasso.predict(X_test_scaled)
```

### Evaluation

```python
lasso_mse = mean_squared_error(y_test, lasso_pred)

lasso_r2 = r2_score(y_test, lasso_pred)
```

### Lasso Objective

```text
Loss = MSE + λ × Σ|coefficient|
```

### Purpose

✔ Reduces overfitting
✔ Can reduce unnecessary feature coefficients
✔ Helps in feature selection

---

# 🔄 PART D — Cross Validation

<div align="center">

<img src="assets/cross-validation.gif" alt="Cross Validation GIF" width="600">

</div>

Cross Validation is used to check whether the model performs consistently on different data splits.

---

## 1️⃣ K-Fold Cross Validation

The project uses:

```text
5-Fold Cross Validation
```

Code:

```python
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kfold_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kfold,
    scoring="r2"
)
```

Output includes:

```text
Individual R² Scores
Mean R² Score
Standard Deviation
```

---

## 2️⃣ Stratified K-Fold

Since this is a regression problem, target values are divided into bins before applying stratification.

```python
y_bins = pd.qcut(
    y_train,
    q=5,
    labels=False,
    duplicates="drop"
)
```

This helps maintain a balanced target distribution across folds.

---

## 📚 Other Validation Methods Used

The project also imports validation techniques such as:

```text
KFold
StratifiedKFold
LeaveOneOut
TimeSeriesSplit
```

These methods help compare different validation strategies.

---

# 🌳 PART E — Decision Tree Regression

<div align="center">

<img src="assets/decision-tree.gif" alt="Decision Tree GIF" width="500">

</div>

Decision Tree Regression learns patterns by splitting the data into different decision regions.

```python
tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

Model training:

```python
tree.fit(X_train, y_train)
```

Prediction:

```python
tree_train_prediction = tree.predict(X_train)

tree_test_prediction = tree.predict(X_test)
```

---

## 📊 Decision Tree Evaluation

The following metrics are calculated:

```text
Training MSE
Testing MSE
Testing R² Score
```

Code:

```python
tree_train_mse = mean_squared_error(
    y_train,
    tree_train_prediction
)

tree_test_mse = mean_squared_error(
    y_test,
    tree_test_prediction
)

tree_r2 = r2_score(
    y_test,
    tree_test_prediction
)
```

---

## 🌱 Tree Complexity Analysis

Different tree depths are tested:

```python
depth_values = [2, 3, 5, 7, 10, None]
```

This helps analyze:

```text
Small Depth  → Possible Underfitting
Optimal Depth → Better Generalization
Large Depth  → Possible Overfitting
```

---

# 🌲 Random Forest Regression

Random Forest combines predictions from multiple Decision Trees.

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤
Tree 4 ─┼──► Final Prediction
Tree 5 ─┤
Tree N ─┘
```

### Why Random Forest?

✔ Uses multiple trees
✔ Reduces overfitting compared with a single tree
✔ Can capture complex relationships
✔ Produces a combined prediction

---

# 🔴 PART F — Support Vector Regression

<div align="center">

<img src="assets/svr.gif" alt="SVR GIF" width="500">

</div>

Support Vector Regression is used with multiple kernels to compare different relationships between features and house prices.

---

## 1️⃣ Linear SVR

```python
svr_linear = make_pipeline(

    StandardScaler(),

    SVR(
        kernel="linear",
        C=100,
        epsilon=0.1
    )
)
```

Training:

```python
svr_linear.fit(
    X_train,
    y_train
)
```

Evaluation:

```python
linear_mse = mean_squared_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)
```

---

## 2️⃣ Polynomial SVR

```python
SVR(
    kernel="poly",
    C=100,
    epsilon=0.1,
    degree=3
)
```

This model is useful when the relationship between variables is more complex and non-linear.

---

## 🧠 SVR Parameters Used

| Parameter | Meaning                                 |
| --------- | --------------------------------------- |
| `kernel`  | Type of relationship used by SVR        |
| `C`       | Controls penalty for errors             |
| `epsilon` | Defines the acceptable error margin     |
| `degree`  | Polynomial degree for polynomial kernel |

---

# 📊 PART G — Model Evaluation

The project evaluates models using multiple metrics.

<div align="center">

<img src="assets/model-comparison.gif" alt="Model Comparison GIF" width="600">

</div>

## 📐 Metrics Used

### 1️⃣ Mean Squared Error — MSE

```text
MSE = Average of Squared Errors
```

Lower MSE generally indicates better prediction accuracy.

---

### 2️⃣ Mean Absolute Error — MAE

```text
MAE = Average Absolute Difference
between Actual and Predicted Values
```

Lower MAE indicates smaller prediction errors.

---

### 3️⃣ Root Mean Squared Error — RMSE

```text
RMSE = √MSE
```

RMSE shows prediction error in the target variable's scale.

---

### 4️⃣ R² Score

```text
R² = Model's ability to explain
variation in the target variable
```

Higher R² generally indicates better model performance.

---

## 🧮 Evaluation Function

The project calculates:

```text
Training MSE
Testing MSE

Training MAE
Testing MAE

Training RMSE
Testing RMSE

Training R²
Testing R²
```

This allows a detailed comparison between training and testing performance.

---

# 🏆 PART H — Best Model Selection

The project automatically identifies the best-performing model based on:

```text
Highest Test R² Score
```

Code:

```python
best_model = results_df.loc[
    results_df["Test R2"].idxmax()
]
```

The selected model displays:

```text
🏆 Model Name
📉 Test MSE
📊 Test MAE
📏 Test RMSE
⭐ Test R² Score
```

---

# 🔍 Regularization Analysis

The project also performs regularization analysis.

The main techniques compared are:

| Model            | Regularization    |
| ---------------- | ----------------- |
| Ridge Regression | L2 Regularization |
| Lasso Regression | L1 Regularization |

This analysis helps understand how regularization affects model performance and coefficients.

---

# 🤖 Machine Learning Models Used

<div align="center">

| No. | Model                   | Type                     |
| --- | ----------------------- | ------------------------ |
| 1️⃣ | Linear Regression       | Linear Model             |
| 2️⃣ | Ridge Regression        | Regularized Linear Model |
| 3️⃣ | Lasso Regression        | Regularized Linear Model |
| 4️⃣ | Decision Tree Regressor | Tree-Based Model         |
| 5️⃣ | Random Forest Regressor | Ensemble Model           |
| 6️⃣ | SVR - Linear Kernel     | Support Vector Model     |
| 7️⃣ | SVR - Polynomial Kernel | Support Vector Model     |

</div>

---

# 📊 Model Comparison Strategy

All models are evaluated using the same important metrics.

```text
                    TRAIN MODELS
                         │
      ┌──────────┬───────┼────────┬──────────┐
      ▼          ▼       ▼        ▼          ▼
    Ridge      Lasso    Tree    Forest       SVR
      │          │       │        │          │
      └──────────┴───────┼────────┴──────────┘
                         ▼
                 CALCULATE METRICS
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
            MSE         MAE         RMSE
                         │
                         ▼
                        R²
                         │
                         ▼
                  🏆 BEST MODEL
```

---

# 🚀 How to Run the Project

## Step 1 — Clone or Download the Project

Download the project files to your computer.

---

## Step 2 — Install Required Libraries

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
```

Or:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Step 3 — Keep Dataset in the Same Folder

Make sure these files are together:

```text
project2(3).ipynb
house_price.csv
```

---

## Step 4 — Open Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
project2(3).ipynb
```

---

## Step 5 — Run All Cells

Run the notebook from top to bottom.

```text
Kernel
   ↓
Restart & Run All
```

The project will:

1. Load the dataset
2. Prepare features and target
3. Split training and testing data
4. Scale features where required
5. Train multiple regression models
6. Perform validation
7. Calculate performance metrics
8. Compare models
9. Select the best model

---

# 📸 Screenshots / Output

You can add your project output screenshots here.

### Dataset Preview

```text
assets/dataset-preview.png
```

### Ridge and Lasso Output

```text
assets/ridge-lasso-output.png
```

### Cross Validation Output

```text
assets/cross-validation-output.png
```

### Decision Tree Output

```text
assets/decision-tree-output.png
```

### SVR Output

```text
assets/svr-output.png
```

### Final Model Comparison

```text
assets/final-model-comparison.png
```

Example:

```html
<img src="assets/final-model-comparison.png"
     alt="Final Model Comparison"
     width="800">
```

---

# 📁 Recommended Assets Folder

To make the GitHub README look attractive, create an `assets` folder and add:

```text
assets/
│
├── house.gif
├── ml-workflow.gif
├── ridge-lasso.gif
├── cross-validation.gif
├── decision-tree.gif
├── random-forest.gif
├── svr.gif
├── model-comparison.gif
│
├── dataset-preview.png
├── ridge-lasso-output.png
├── cross-validation-output.png
├── decision-tree-output.png
├── svr-output.png
└── final-model-comparison.png
```

---

# 💡 Key Learning Outcomes

After completing this project, the following Machine Learning concepts are covered:

* [x] Regression Fundamentals
* [x] Data Loading with Pandas
* [x] Feature Selection
* [x] Target Variable Selection
* [x] Train-Test Split
* [x] StandardScaler
* [x] Ridge Regression
* [x] Lasso Regression
* [x] Regularization
* [x] RidgeCV
* [x] LassoCV
* [x] GridSearchCV
* [x] K-Fold Cross Validation
* [x] Stratified K-Fold
* [x] Leave-One-Out Validation
* [x] Time Series Split
* [x] Decision Tree Regression
* [x] Tree Complexity Analysis
* [x] Random Forest Regression
* [x] Support Vector Regression
* [x] Linear SVR
* [x] Polynomial SVR
* [x] MSE
* [x] MAE
* [x] RMSE
* [x] R² Score
* [x] Model Comparison
* [x] Best Model Selection

---

# 🎯 Project Objective

> The main objective of this project is to build and compare multiple Machine Learning Regression models for House Price Prediction and select the model with the strongest testing performance based on evaluation metrics.

---

# 🏆 Final Result

```text
              MULTIPLE ML MODELS
                       │
                       ▼
               PERFORMANCE CHECK
                       │
                       ▼
        MSE + MAE + RMSE + R² SCORE
                       │
                       ▼
             COMPARE ALL MODELS
                       │
                       ▼
          SELECT HIGHEST TEST R² MODEL
                       │
                       ▼
                  🏆 BEST MODEL
```

---

<div align="center">

# 🏠 House Price Prediction Project

### 📊 Machine Learning | 🤖 Regression | 📈 Model Evaluation

<br>

**Made with ❤️ using Python & Machine Learning**

⭐ If you found this project useful, don't forget to give it a Star!

</div>
::: ​​
