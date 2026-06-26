# Personalized-Customer-Retention-using-Causal-Uplift-Modeling-and-CatBoost-X-Learner
# Customer Retention Prediction using Uplift Modeling (XLearner + CatBoost)

## Overview

This project focuses on **customer retention using Uplift Modeling** rather than traditional churn prediction. Instead of predicting *who will churn*, the model predicts **which customers are most likely to respond positively to a retention treatment**, allowing businesses to target only the customers who generate the highest business value.

The project is built using the **IBM Telco Customer Churn Dataset**, with **CatBoost as the base learner** and **XLearner** as the final uplift model. The complete solution is deployed using **Streamlit** for interactive customer analysis.

---

# Problem Statement

Traditional churn prediction models identify customers who are likely to leave but fail to answer an important business question:

> **Which customers should actually receive a retention offer?**

Providing discounts or incentives to every at-risk customer increases marketing costs and often reduces profit. This project solves that problem using **Uplift Modeling**, which estimates the causal effect of treatment on customer retention.

---

# Dataset

* IBM Telco Customer Churn Dataset
* CSV format
* Includes demographic, service, billing, and subscription details

---

# Project Workflow

## 1. Data Preprocessing

* Loaded IBM Telco dataset
* Removed duplicate records
* Handled missing values
* Performed feature preprocessing
* Encoded categorical variables
* Selected relevant features
* Removed highly correlated features

---

## 2. Exploratory Data Analysis (EDA)

Performed extensive EDA to understand customer behavior.

Analysis included:

* Churn distribution
* Feature importance visualization
* Correlation analysis
* Customer demographic analysis
* Service usage analysis
* Contract and payment behavior

---

## 3. Treatment & Outcome Construction

Created uplift learning target variables:

### Y0 (Control Outcome)

Represents customer retention **without treatment**.

* Y0 = 1 → Customer retained without treatment
* Y0 = 0 → Customer churned without treatment

### Y1 (Treatment Outcome)

Represents customer retention **after treatment**.

* Y1 = 1 → Customer retained after treatment
* Y1 = 0 → Customer churned despite treatment

Using Y0 and Y1, uplift labels were generated for causal learning.

---

## 4. Uplift Modeling

Used **CatBoost** as the base learner and compared multiple meta-learning approaches:

* S-Learner
* T-Learner
* X-Learner

Model comparison was based on:

* Correlation score
* Mean Absolute Error (MAE)
* Overall prediction quality

**Best Model:** XLearner with CatBoost

---

## 5. Model Serialization

Saved the trained XLearner pipeline using Pickle for deployment.

---

## 6. SHAP Explainability

Performed SHAP analysis to identify features contributing most to uplift.

The explainability module helps understand:

* Which features increase treatment effectiveness
* Which customers benefit the most from intervention

---

## 7. Customer Segmentation

Customers are categorized based on predicted uplift:

### Persuadables

Customers likely to stay **only if treatment is given**.

High priority for retention campaigns.

### Sure Things

Customers who stay even without treatment.

No incentive required.

### Sleeping Dogs

Customers who may react negatively to treatment.

Avoid targeting them.

### Lost Causes

Customers unlikely to stay even after treatment.

Low priority for retention efforts.

---

## 8. ROI Analysis

Estimated business impact using:

* Treatment Cost = ₹100
* Revenue per retained customer = ₹1000

Calculated:

* Expected revenue
* Treatment cost
* Net profit
* Return on Investment (ROI)

Average ROI obtained:

**≈ ₹80 per customer**

---

# Streamlit Dashboard

The interactive dashboard includes:

## Executive Summary

* Total Customers
* Average Uplift (~18%)
* Average ROI
* Customer Statistics

---

## Customer Segmentation Dashboard

Visualizes:

* Persuadables
* Sure Things
* Sleeping Dogs
* Lost Causes

---

## ROI Dashboard

Displays expected business returns after applying treatment.

---

## Individual Customer Analysis

Users can enter a Customer ID to view:

* Predicted uplift
* Customer segment
* Priority level
* Expected ROI
* Treatment recommendation

---

## SHAP Feature Importance

Displays the top features influencing uplift predictions.

---

## Uplift Distribution

Shows how uplift scores are distributed across the customer population.

---

# Inference Pipeline


User
   │
   ▼
Streamlit Dashboard
   │
   ▼
Inference Pipeline
   │
   ├── Load Pickle Model
   ├── Load Preprocessor
   ├── Generate Predictions
   ├── Calculate Uplift
   ├── Customer Segmentation
   ├── ROI Calculation
   ├── SHAP Feature Importance
   ▼
Results returned to Streamlit Dashboard


---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* XLearner
* SHAP
* Matplotlib
* Plotly
* Streamlit
* Pickle

---

# Key Results

* Built an end-to-end uplift modeling pipeline.
* Compared S-Learner, T-Learner, and X-Learner.
* Selected XLearner with CatBoost as the best-performing model.
* Achieved an average uplift of approximately **18%**.
* Estimated an average ROI of around **₹80 per customer**.
* Developed an interactive Streamlit dashboard for customer-level decision support.
* Integrated SHAP explainability for transparent model interpretation.

---

# Future Improvements

* Real-time API deployment
* A/B testing integration
* Dynamic treatment cost optimization
* Multi-treatment uplift modeling
* Automated retraining pipeline
