# 💰 Employee Salary Predictor

An end-to-end Machine Learning application that predicts an employee's estimated salary in USD based on professional, employment, location, remote-work, and company-related attributes.

The project combines multiple regression algorithms with automated preprocessing and an interactive Streamlit dashboard to provide salary predictions, model comparison, feature importance, salary analytics, and downloadable prediction reports.

---

## 🎯 Project Objective

The objective of this project is to develop a practical Machine Learning system capable of estimating employee salaries using historical data.

The system demonstrates a complete ML workflow:

**Data → Preprocessing → Feature Encoding → Model Training → Evaluation → Model Selection → Prediction → Interactive Dashboard**

---

## ✨ Key Features

* 💰 Employee salary prediction
* 🤖 Multiple Machine Learning regression models
* 🌲 Random Forest Regressor
* 📈 Gradient Boosting Regressor
* 🌳 Extra Trees Regressor
* 🗳️ Voting Regressor ensemble
* ⚙️ Automated data preprocessing
* 🔤 One-Hot Encoding for categorical variables
* 📊 Model performance comparison
* 🧠 Feature importance analysis
* 📈 Salary analytics
* 🕘 Prediction history
* 📥 Downloadable prediction report
* 🖥️ Interactive Streamlit dashboard
* 🛡️ Input validation and error handling

---

## 🧠 Machine Learning Models

The project evaluates several regression algorithms:

### 1. Random Forest Regressor

An ensemble of decision trees that combines multiple predictions to improve generalization and reduce overfitting.

### 2. Gradient Boosting Regressor

Builds an ensemble sequentially, where each new model attempts to improve the errors of previous models.

### 3. Extra Trees Regressor

Uses randomized decision-tree construction to create a diverse ensemble of trees.

### 4. Voting Regressor

Combines predictions from multiple regression models to produce a final ensemble prediction.

Since salary is a continuous numerical target, the project uses **regression**, specifically `VotingRegressor`, rather than `VotingClassifier`.

---

## 📊 Dataset

The project uses the **Data Science Salaries 2023** dataset.

The dataset contains information including:

* Work year
* Experience level
* Employment type
* Job title
* Salary
* Salary currency
* Salary in USD
* Employee residence
* Remote work ratio
* Company location
* Company size

### Target Variable

```text
salary_in_usd
```

The original `salary` and `salary_currency` columns are removed during preprocessing to avoid unnecessary target-related information.

---

## ⚙️ Technology Stack

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | Programming language        |
| Pandas       | Data processing             |
| NumPy        | Numerical operations        |
| Scikit-learn | Machine Learning            |
| Matplotlib   | Visualization               |
| Seaborn      | Data visualization          |
| Joblib       | Model serialization         |
| Streamlit    | Interactive web application |
| Git & GitHub | Version control             |

---

## 🔄 Machine Learning Workflow

```text
                Dataset
                   │
                   ▼
           Data Exploration
                   │
                   ▼
        Data Cleaning & Preparation
                   │
                   ▼
        Feature / Target Separation
                   │
                   ▼
       Categorical Feature Encoding
                   │
                   ▼
            Train / Test Split
                   │
                   ▼
        ┌─────────────────────────┐
        │     Model Training      │
        │                         │
        │ Random Forest           │
        │ Gradient Boosting       │
        │ Extra Trees             │
        │ Voting Regressor        │
        └─────────────────────────┘
                   │
                   ▼
          Model Evaluation
                   │
                   ▼
            Best Model
                   │
                   ▼
          salary_model.pkl
                   │
                   ▼
        Streamlit Prediction App
```

---

## 📏 Model Evaluation

The models are evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted salaries.

### R
