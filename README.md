# British Airways Customer Booking Prediction ✈️

A Machine Learning project developed as part of **Task 2 of the British Airways Data Science Job Simulation on Forage**.

The project focuses on predicting whether a customer will complete a flight booking and identifying the key factors that influence customer booking decisions.

---

## 📌 Project Overview

British Airways wants to better understand customer booking behavior and identify factors that influence whether a customer completes a flight booking.

Using a dataset containing **50,000 customer booking records**, this project develops a Machine Learning pipeline to:

- Analyze customer booking behavior through Exploratory Data Analysis (EDA)
- Perform data preprocessing and feature engineering
- Create domain-specific features related to customer intent and booking patterns
- Train a Machine Learning classification model
- Evaluate model performance using multiple metrics
- Identify the most influential features affecting booking completion
- Translate Machine Learning findings into actionable business insights

### 🎯 Target Variable

The target variable is:

`booking_complete`

Where:

- `0` → Booking was not completed
- `1` → Booking was completed

---

## 📂 Repository Structure

```text
british-airways-data-science/
│
├── data/
│   └── customer_booking.csv
│       └── Dataset containing 50,000 customer flight booking records
│
├── notebooks/
│   └── customer_booking_prediction.ipynb
│       └── EDA, feature engineering, model training and evaluation
│
├── visuals/
│   └── feature_importance.png
│       └── Random Forest feature importance visualization
│
├── deliverables/
│   └── British_Airways_Task2_Executive_Summary.pptx
│       └── Executive presentation summarizing findings and recommendations
│
├── requirements.txt
│   └── Python package dependencies
│
└── README.md
    └── Project documentation