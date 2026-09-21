# British Airways Customer Booking Prediction

This repository contains the Machine Learning solution for **Task 2** of the **British Airways Data Science Job Simulation** on Forage. The goal of this project is to build a predictive machine learning model to uncover customer booking patterns and determine the key factors that drive flight booking completions.

---

## 📂 Repository Structure

```text
british-airways-data-science/
├── data/
│   └── customer_booking.csv                  # Raw dataset containing 50,000 customer booking sessions
├── notebooks/
│   └── customer_booking_prediction.ipynb     # End-to-end Jupyter Notebook for data cleaning, EDA, and modeling
├── visuals/
│   └── feature_importance.png                # Visualization showing relative feature importance drivers
├── deliverables/
│   └── British_Airways_Task2_Executive_Summary.pptx # Non-technical executive presentation for stakeholders
├── requirements.txt                          # Python dependencies required to run the notebook
└── README.md                                 # High-level overview and documentation
```

## 📌 Business Context & Objective

British Airways wants to increase conversion rates across its direct booking channels. When customers browse flight schedules, choose travel add-ons, or select routes, their digital interaction patterns provide strong signals regarding whether they will complete a reservation.

The primary objectives of this project are:

1. **Predictive Analytics:** Build a binary classification model that accurately predicts whether a customer session ends in a completed booking (`booking_complete`).

2. **Commercial Insights:** Uncover the primary behavioral drivers and customer preferences (e.g., lead times, extra services, trip duration) that influence purchase decisions.

3. **Strategic Recommendations:** Translate model insights into actionable strategies for marketing, pricing, and product design teams to boost overall booking conversions.

## 🛠️ Methodological Approach

### 1. Data Cleaning & Exploratory Data Analysis (EDA)

- **Data Integrity & Consistency:** Inspected dataset quality across 50,000 records to identify missing values, ensure consistent variable encoding, and convert textual categorical variables into numerical formats.

- **Handling Class Imbalance:** Analyzed the distribution of the target variable, identifying a natural class imbalance (~15% completed bookings vs. ~85% non-completions).

### 2. Feature Engineering

Created domain-specific features to capture nuanced customer behavior:

- **Planning & Timing Indicators:** Derived customer lead-time ratios and behavioral group-booking dynamics.

- **Aggregated Demand Metrics:** Mapped historical route frequencies and origin market demand patterns to contextualize high-volume customer segments.

- **Service Bundling:** Analyzed combinations of ancillary add-ons (extra baggage, preferred seat selection, in-flight meals) to assess purchase intent.

### 3. Model Building & Validation

- **Algorithm Selection:** Selected a **Random Forest Classifier** due to its robust handling of non-linear customer behaviors, feature interactions, and tabular data.

- **Validation Strategy:** Employed **5-Fold Stratified Cross-Validation** to ensure the model generalizes well across imbalanced dataset splits without overfitting.

- **Balanced Class Weights:** Adjusted decision tree weighting during training to handle class imbalance without sacrificing predictive reliability on successful bookings.

## 💡 Key Business Drivers & Strategic Insights

The feature importance analysis highlighted four primary operational levers that dictate customer booking completions:

1. **Purchase Lead Time (`purchase_lead`):** The advance booking duration is the single strongest indicator of purchase intent. Customers who reserve flights far in advance demonstrate significantly different purchasing patterns than last-minute browsers.

2. **Route & Origin Demand (`route_freq`, `booking_origin_freq`):** Geographic market location and route volume density heavily impact conversion velocity. High-density origin markets reflect strong organic customer demand.

3. **Trip Duration (`length_of_stay`):** The length of stay correlates strongly with booking conversion, as long-haul and holiday travel decisions involve distinct evaluation windows compared to short trips.

4. **Group Lead Dynamics (`lead_time_per_passenger`):** Planning timeline behavior varies proportionally with party size, signaling key differences between solo business travelers and family holiday bookings.

## 🛠️ Tech Stack & Environment

- **Language:** Python 3.13
- **Data Analysis & Processing:** `pandas`, `numpy`
- **Machine Learning & Evaluation:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Visual Studio Code / Jupyter Notebooks

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone [https://github.com/paraskosambe-web/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME](https://github.com/paraskosambe-web/British-Airways---Data-Science-Job-Simulation.git)
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Execute the Python code

Launch VS Code or Jupyter Lab, open `task1.py` and `task2.py` , and run all cells to perform data processing, train the Random Forest model, and generate visual outputs.

## 📋 Project Deliverables

Contains the final result
