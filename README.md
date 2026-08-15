<p align="center">
  <img src="https://github.com/pmanan2031-web/ML.PROJECT.2/blob/main/image/house_price_prediction.png?raw=true"
       alt="House Price Prediction Project"
       width="100%">
</p>

<p align="center">
  <img src="./image/house_price_prediction.png"
       alt="House Price Prediction Project"
       width="100%">
</p>
# 🏠 House Price Prediction

### Predicting House Prices Using Machine Learning

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square\&logo=scikitlearn\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

<br>

> An end-to-end Machine Learning project that prepares house price data, trains multiple regression models, evaluates their performance, and selects the best model.

</div>

---

## 📌 Table of Contents

* [About The Project](#-about-the-project)
* [Project Preview](#-project-preview)
* [Workflow](#-workflow)
* [Features](#-features)
* [Technologies Used](#️-technologies-used)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Project Structure](#-project-structure)
* [How To Run](#-how-to-run)
* [Learning Outcomes](#-learning-outcomes)

---

# 📖 About The Project

House price prediction is a **Regression Machine Learning problem**.

In this project, the dataset is prepared and multiple Machine Learning models are trained to predict **house prices**. Different models are then compared using regression evaluation metrics to identify the best-performing model.

### 🎯 Main Goal

```text
Predict House Price → Train Multiple Models → Compare Performance → Select Best Model
```

---

# 🎬 Project Preview

<div align="center">

### 🚀 Complete Project Demo

<img src="./images/project-demo.gif" width="850" alt="Project Demo"/>

</div>

> 📌 Add your project GIF inside the `images` folder with the name `project-demo.gif`.

---

# 🔄 Workflow

<div align="center">

```text
 ┌───────────────────┐
 │   📂 Load Data     │
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ 🧹 Data Preparation│
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ ✂️ Train-Test Split│
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ ⚙️ Feature Scaling │
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ 🤖 Train Models   │
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ 📊 Evaluate Models │
 └─────────┬─────────┘
           ↓
 ┌───────────────────┐
 │ 🏆 Best Model      │
 └───────────────────┘
```

<br>

<img src="./images/workflow.gif" width="800" alt="Project Workflow"/>

</div>

---

# ✨ Features

<table>
<tr>
<td width="50%">

### 🧹 Data Preparation

* Clean column names
* Select features and target
* Remove unnecessary columns
* Prepare data for ML

</td>

<td width="50%">

### ⚙️ Feature Processing

* Train-Test Split
* Feature Scaling
* StandardScaler

</td>
</tr>

<tr>
<td width="50%">

### 🤖 Model Training

* Multiple Regression Models
* Tree-Based Models
* Support Vector Regression

</td>

<td width="50%">

### 📊 Model Comparison

* MSE
* MAE
* RMSE
* R² Score

</td>
</tr>
</table>

---

# 🎞️ Data Preparation

<div align="center">

<img src="./images/data-preparation.gif" width="750" alt="Data Preparation"/>

</div>

### Basic Steps

```text
📂 Load Dataset
      ↓
🧹 Clean Data
      ↓
🎯 Select X and y
      ↓
✂️ Split Data
      ↓
⚙️ Scale Features
```

---

# 🛠️ Technologies Used

<div align="center">

| Technology          | Purpose                   |
| :------------------ | :------------------------ |
| 🐍 Python           | Main Programming Language |
| 🐼 Pandas           | Data Analysis             |
| 🔢 NumPy            | Numerical Operations      |
| 🤖 Scikit-learn     | Machine Learning          |
| 📊 Matplotlib       | Data Visualization        |
| 📓 Jupyter Notebook | Project Development       |

</div>

---

# 🤖 Machine Learning Models

<div align="center">

<img src="./images/model-training.gif" width="800" alt="Model Training"/>

</div>

The following models are used and compared in this project:

<table>
<tr>
<td align="center" width="33%">

### 🔵 Ridge

Regularized Regression

</td>
<td align="center" width="33%">

### 🟠 Lasso

Feature Regularization

</td>
<td align="center" width="33%">

### 🌱 Decision Tree

Tree-Based Model

</td>
</tr>

<tr>
<td align="center" width="33%">

### 🌲 Random Forest

Ensemble Learning

</td>
<td align="center" width="33%">

### 📈 SVR

Linear Kernel

</td>
<td align="center" width="33%">

### 🔴 SVR

Poly & RBF Kernels

</td>
</tr>
</table>

---

# 🔄 Cross Validation

Multiple validation methods are used to test model performance.

<div align="center">

| Method               | Description                           |
| :------------------- | :------------------------------------ |
| 🔹 K-Fold            | Data is divided into multiple folds   |
| 🔹 Stratified K-Fold | Balanced validation using target bins |
| 🔹 Leave-One-Out     | One sample used for validation        |
| 🔹 Time Series Split | Sequential data validation            |

<br>

<img src="./images/cross-validation.gif" width="750" alt="Cross Validation"/>

</div>

---

# ⚙️ Hyperparameter Tuning

Different parameters are tested to improve model performance.

```text
Alpha  •  C  •  Gamma  •  Epsilon  •  Tree Depth
```

<div align="center">

<img src="./images/tuning.gif" width="750" alt="Hyperparameter Tuning"/>

</div>

---

# 📊 Model Evaluation

Each model is evaluated using the following metrics:

<div align="center">

|    Metric   | Purpose                             |
| :---------: | :---------------------------------- |
|    📉 MSE   | Measures squared prediction error   |
|    📏 MAE   | Measures average absolute error     |
|   📊 RMSE   | Measures prediction error magnitude |
| 🎯 R² Score | Measures overall model performance  |

</div>


# 🏆 Best Model Selection

<div align="center">

<img src="./images/best-model.gif" width="750" alt="Best Model Selection"/>

</div>

All trained models are compared based on their performance.

```text
🤖 All Models
      ↓
📊 Calculate Metrics
      ↓
📈 Compare Results
      ↓
🏆 Best Performing Model
```

The model with the strongest test performance is selected as the **Best Model**.

---

# 📁 Project Structure

```text
House-Price-Prediction/
│
├── 📂 images/
│   ├── banner.gif
│   ├── project-demo.gif
│   ├── workflow.gif
│   ├── data-preparation.gif
│   ├── model-training.gif
│   ├── cross-validation.gif
│   ├── tuning.gif
│   ├── best-model.gif
│   ├── model_comparison.png
│   └── r2_comparison.png
│
├── 📓 project2(2).ipynb
├── 📄 house_price.csv
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 🚀 How To Run

### 1️⃣ Clone the Repository

```bash
git clone YOUR_REPOSITORY_LINK
```

### 2️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3️⃣ Open Jupyter Notebook

```bash
jupyter notebook
```

### 4️⃣ Run the Project

Open:

```text
project2(2).ipynb
```

Run all cells from top to bottom. 🚀

---

# 📚 Learning Outcomes

Through this project, the following concepts are covered:

<div align="center">

`Data Preparation` • `Feature Scaling` • `Regression`

`Cross Validation` • `Ridge` • `Lasso`

`Decision Tree` • `Random Forest` • `SVR`

`Hyperparameter Tuning` • `Model Evaluation` • `Model Comparison`

</div>

---

# 🌟 Project Highlights

<div align="center">

|        🧹        |        🤖       |        🔄        |           📊           |     🏆     |
| :--------------: | :-------------: | :--------------: | :--------------------: | :--------: |
| Data Preparation | Multiple Models | Cross Validation | Performance Comparison | Best Model |

</div>

---

<div align="center">

## ⭐ If You Like This Project

### Don't forget to give this repository a ⭐

<br>

**Made with ❤️ using Python and Machine Learning**

</div>
