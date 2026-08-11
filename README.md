<div align="center" id="top">

🏠 House Price Prediction using Machine Learning

Regularized Linear Models • Cross-Validation • Decision Trees • Random Forest • SVR

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=plotly&logoColor=white" alt="Matplotlib">
</p>

<br>

<a href="./project2.ipynb">
  <img src="https://img.shields.io/badge/📓%20OPEN%20NOTEBOOK-project2.ipynb-2ea44f?style=for-the-badge" alt="Open Notebook">
</a>
&nbsp;
<a href="./house_price.csv">
  <img src="https://img.shields.io/badge/📊%20DATASET-house__price.csv-0969DA?style=for-the-badge" alt="Dataset">
</a>
&nbsp;
<a href="#project-workflow">
  <img src="https://img.shields.io/badge/⚙️%20WORKFLOW-view%20project-6f42c1?style=for-the-badge" alt="Workflow">
</a>

<br><br>

<img src="./assets/house_price_model_comparison.gif" alt="Animated House Price Model Comparison" width="900">

</div>

📌 Project Overview

This project builds and compares multiple machine learning regression models for predicting House Price from real-estate property information.

The project follows a complete modelling workflow:

Dataset Understanding → Preprocessing → Ridge & Lasso → Cross-Validation → Decision Tree → Random Forest → SVR → Model Comparison

The work is implemented in the uploaded Jupyter Notebook:

project2.ipynb

🔘 Quick Navigation

<table>
<tr>
<td align="center"><a href="#project-overview">🏠 Overview</a></td>
<td align="center"><a href="#dataset">📊 Dataset</a></td>
<td align="center"><a href="#project-workflow">⚙️ Workflow</a></td>
<td align="center"><a href="#models-used">🤖 Models</a></td>
<td align="center"><a href="#results">📈 Results</a></td>
</tr>
<tr>
<td align="center"><a href="#cross-validation">🔄 Cross-Validation</a></td>
<td align="center"><a href="#tree-based-models">🌳 Tree Models</a></td>
<td align="center"><a href="#support-vector-regression">📐 SVR</a></td>
<td align="center"><a href="#installation">💻 Installation</a></td>
<td align="center"><a href="#future-improvements">🚀 Future Work</a></td>
</tr>
</table>

🎯 Project Objectives

Understand and prepare a real-estate dataset.

Identify features and the target variable.

Split the data into training and testing sets.

Apply feature scaling where required.

Implement Ridge Regression (L2).

Implement Lasso Regression (L1).

Tune regularization strength (alpha) using cross-validation.

Compare multiple cross-validation strategies.

Build a Decision Tree Regressor.

Control tree complexity using max_depth and min_samples_split.

Build a Random Forest Regressor.

Implement SVR using Linear, Polynomial, and RBF kernels.

Tune SVR hyperparameters C, gamma, and epsilon.

Compare model performance using regression metrics.

Identify possible overfitting and underfitting.

📊 Dataset

The notebook uses:

File: house_price.csv

Rows: 3,800

Columns: 12

Target Variable

houseprice

Features Used

Feature

Description

property_id

Property identifier

area_sqft

Property area in square feet

bedrooms

Number of bedrooms

bathrooms

Number of bathrooms

location_score

Location-based numerical score

property_age

Age of the property

distance_city_km

Distance from city

near_school

School proximity indicator

near_metro

Metro proximity indicator

crime_rate_index

Crime-rate indicator

The notebook excludes sale_date from the final feature matrix before modelling.

Dataset note: The notebook expects house_price.csv in the project root. If you are cloning this repository, keep the CSV beside project2.ipynb.

⚙️ Project Workflow

                    🏠 REAL ESTATE DATA
                            │
                            ▼
                 📊 Dataset Understanding
                            │
                            ▼
                    🧹 Data Preparation
                            │
                            ▼
                   ✂️ Train-Test Split
                            │
                            ▼
                    📏 Feature Scaling
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     📉 Ridge/Lasso    🔄 Cross-Validation   🌳 Tree Models
          │                 │                 │
          │                 │                 ├── Decision Tree
          │                 │                 └── Random Forest
          │                 │
          └─────────────────┼─────────────────┘
                            ▼
                         📐 SVR
                    ├── Linear
                    ├── Polynomial
                    └── RBF
                            │
                            ▼
                  📈 MODEL COMPARISON
                            │
                            ▼
                   🏆 BEST MODEL

🤖 Models Used

1. Ridge Regression — L2

Ridge adds an L2 regularization penalty to reduce overly large coefficients.

Ridge(alpha=1)

The notebook also uses RidgeCV to select a suitable alpha.

2. Lasso Regression — L1

Lasso applies L1 regularization and can shrink some coefficients toward zero.

Lasso(alpha=1)

The notebook also uses LassoCV for alpha selection.

3. Decision Tree Regression

DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

Tree complexity is tested with different max_depth values.

4. Random Forest Regression

RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

Random Forest combines multiple decision trees to improve generalization.

5. Support Vector Regression

Three kernels are tested:

Linear

Polynomial

RBF

Example:

SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

🔄 Cross-Validation

The project compares:

Strategy

Purpose

K-Fold

Splits the training data into multiple folds

Stratified K-Fold

Bins house prices before creating stratified folds

Leave-One-Out

Uses one observation for validation at a time

Time Series Split

Uses sequential training/validation splits

Executed notebook results

Method

Mean R²

Std R²

K-Fold

0.915272

0.004385

Stratified K-Fold

0.915552

0.004297

Time Series

0.914754

0.005019

The notebook also reports a LOOCV Mean MSE of approximately:

5.8769 × 10¹²

🌳 Tree-Based Models

Decision Tree

Executed test performance:

MSE: 9.3858 × 10¹²

R²: 0.883457

Random Forest

Executed test performance:

MSE: 5.8866 × 10¹²

R²: 0.926907

Random Forest performs better than the single Decision Tree in the executed notebook results.

📐 Support Vector Regression

The notebook tests three kernels:

SVR Model

Test MSE

Test R²

Linear

7.2618 × 10¹³

0.098304

Polynomial

8.0257 × 10¹³

0.003452

RBF

8.0566 × 10¹³

-0.000388

The notebook's tested SVR parameter combinations selected:

C        = 100
gamma    = scale
epsilon  = 0.1

📈 Results

Regularized Linear Models

Model

Test MSE

Test R²

Ridge

6.5483 × 10¹²

0.918691

Lasso

6.5471 × 10¹²

0.918705

Best regularized linear model in the executed notebook:

Lasso — R² ≈ 0.918705

Overall Models Reported by the Notebook

Model

Test MSE

Test R²

Ridge

6.5483 × 10¹²

0.918691

Lasso

6.5471 × 10¹²

0.918705

Decision Tree

9.3858 × 10¹²

0.883457

Random Forest

5.8866 × 10¹²

0.926907

SVR Linear

7.2618 × 10¹³

0.098304

SVR Polynomial

8.0257 × 10¹³

0.003452

SVR RBF

8.0566 × 10¹³

-0.000388

🏆 Best Reported Model

Random Forest Regression

Test MSE: 5.8866 × 10¹²

Test R²: 0.926907

📉 Overfitting / Underfitting Analysis

The project compares training and testing performance to identify generalization behaviour.

Decision Tree

The executed notebook reports:

Training MSE: 7.5030 × 10¹²

Testing MSE: 9.3858 × 10¹²

Random Forest

The executed notebook reports:

Training MSE: 1.7739 × 10¹²

Testing MSE: 5.8866 × 10¹²

The gap between training and testing performance should be considered when evaluating model complexity and generalization.

🧪 Evaluation Metrics

The project is designed around common regression metrics:

Mean Squared Error — MSE

Measures the average squared prediction error.

Mean Absolute Error — MAE

Measures the average absolute prediction error.

Root Mean Squared Error — RMSE

RMSE = √MSE

R² Score

Measures the proportion of target variance explained by the model.

Current notebook status: The executed Part C–F cells report MSE and R². Part G is reserved for the complete MSE, MAE, RMSE, R² comparison.

🛠️ Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Scikit-learn

Main Scikit-learn components

train_test_split
StandardScaler
Ridge
Lasso
RidgeCV
LassoCV
KFold
StratifiedKFold
LeaveOneOut
TimeSeriesSplit
DecisionTreeRegressor
RandomForestRegressor
SVR

📁 Project Structure

House-Price-Prediction/
│
├── project2.ipynb
├── house_price.csv
├── README.md
├── requirements.txt
│
└── assets/
    └── house_price_model_comparison.gif

💻 Installation

1. Clone the repository

git clone <YOUR-GITHUB-REPOSITORY-URL>

2. Open the project folder

cd House-Price-Prediction

3. Install dependencies

pip install -r requirements.txt

4. Start Jupyter Notebook

jupyter notebook

5. Open

project2.ipynb

6. Run the notebook

Run the cells from Part B → Part C → Part D → Part E → Part F → Part G in order.

▶️ How to Use

Keep house_price.csv in the same folder as project2.ipynb.

Open project2.ipynb.

Run the import and dataset cells.

Run Part B for preprocessing.

Run Part C for Ridge and Lasso.

Run Part D for cross-validation.

Run Part E for Decision Tree and Random Forest.

Run Part F for SVR.

Run Part G for final model evaluation.

🚀 Future Improvements

Add complete Part G evaluation with MAE and RMSE.

Add feature importance visualization for Random Forest.

Add residual plots.

Add predicted-vs-actual plots.

Tune Random Forest hyperparameters using GridSearchCV.

Improve SVR performance with better feature scaling and parameter search.

Add an interactive house-price prediction interface.

👨‍💻 Project Author

Manan

Machine Learning / Data Science Project

📚 Project Highlights

✅ Real-estate regression dataset
✅ 3,800 observations
✅ Train-test split
✅ Feature scaling
✅ Ridge Regression
✅ Lasso Regression
✅ Alpha tuning
✅ K-Fold Cross-Validation
✅ Stratified K-Fold
✅ Leave-One-Out CV
✅ Time Series Split
✅ Decision Tree Regression
✅ Random Forest Regression
✅ SVR Linear / Polynomial / RBF
✅ SVR hyperparameter tuning
✅ Model comparison

<div align="center">

⭐ If this project helped you, consider giving the repository a star!

<a href="#top">
  <img src="https://img.shields.io/badge/⬆️%20BACK%20TO%20TOP-000000?style=for-the-badge" alt="Back to top">
</a>

</div>
