import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
# Load dataset
df = pd.read_csv("C:\\Users\\Paras\\Downloads\\customer_booking.csv", encoding="ISO-8859-1")

# Clean flight_day to numeric order
mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}
df["flight_day"] = df["flight_day"].map(mapping)

# Create is_weekend column (Fix for KeyError)
df["is_weekend"] = df["flight_day"].apply(lambda x: 1 if x in [6, 7] else 0)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
# Frequency encoding for high-cardinality categorical variables
for col in ["booking_origin", "route"]:
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)

# Total extra services requested
df["total_extras"] = (
    df["wants_extra_baggage"]
    + df["wants_preferred_seat"]
    + df["wants_in_flight_meals"]
)

# Lead time per passenger ratio
df["lead_time_per_passenger"] = df["purchase_lead"] / df["num_passengers"]

# Define feature columns and target variable
feature_cols = [
    "num_passengers",
    "purchase_lead",
    "length_of_stay",
    "flight_hour",
    "flight_day",
    "is_weekend",
    "wants_extra_baggage",
    "wants_preferred_seat",
    "wants_in_flight_meals",
    "flight_duration",
    "booking_origin_freq",
    "route_freq",
    "total_extras",
    "lead_time_per_passenger",
]

X = df[feature_cols]
y = df["booking_complete"]

# One-Hot Encoding for any remaining categorical features if present
X = pd.get_dummies(X, drop_first=True)

# Train/Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 3. MODEL TRAINING & CROSS-VALIDATION
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    rf_model.fit(X_cv_train, y_cv_train)
    preds_prob = rf_model.predict_proba(X_cv_val)[:, 1]
    cv_scores.append(roc_auc_score(y_cv_val, preds_prob))

print(f"5-Fold CV ROC-AUC Score: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

# Fit model on entire training set
rf_model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION ON TEST SET
# ==========================================
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Test Set Evaluation ---")
print(f"Test ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision:         {precision_score(y_test, y_pred):.3f}")
print(f"Recall:            {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:          {f1_score(y_test, y_pred):.3f}")

# ==========================================
# 5. FEATURE IMPORTANCE VISUALIZATION
# ==========================================
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh", color="#075aaa")  # BA Blue
plt.title("British Airways Booking Model - Feature Importance", fontsize=14, pad=15)
plt.xlabel("Relative Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()