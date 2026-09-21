import os
import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================
file_path = r"C:\Users\Paras\Downloads\British Airways Summer Schedule Dataset - Forage Data Science Task 1 (1).xlsx"

# Read the excel file
df = pd.read_excel(file_path, sheet_name="british_airways_schedule_summer")

print(f"[Info] Loaded {len(df):,} flight records.")

# ==========================================
# 2. FEATURE CALCULATIONS
# ==========================================
# Create Operational Tier (e.g., LONG - Morning, SHORT - Evening)
df["OPERATIONAL_TIER"] = (
    df["HAUL"].astype(str) + " - " + df["TIME_OF_DAY"].astype(str)
)

# Calculate seat capacity per flight
df["TOTAL_SEATS"] = (
    df["FIRST_CLASS_SEATS"]
    + df["BUSINESS_CLASS_SEATS"]
    + df["ECONOMY_SEATS"]
)

# Calculate total lounge-eligible passengers per flight across Tier 1, Tier 2, and Tier 3
df["TOTAL_ELIGIBLE_PAX"] = (
    df["TIER1_ELIGIBLE_PAX"]
    + df["TIER2_ELIGIBLE_PAX"]
    + df["TIER3_ELIGIBLE_PAX"]
)

# ==========================================
# 3. AGGREGATE BY OPERATIONAL TIER
# ==========================================
summary_df = (
    df.groupby("OPERATIONAL_TIER")
    .agg(
        TOTAL_FLIGHTS=("FLIGHT_NO", "count"),
        TOTAL_SEATS=("TOTAL_SEATS", "sum"),
        TIER1_PAX=("TIER1_ELIGIBLE_PAX", "sum"),
        TIER2_PAX=("TIER2_ELIGIBLE_PAX", "sum"),
        TIER3_PAX=("TIER3_ELIGIBLE_PAX", "sum"),
        TOTAL_ELIGIBLE_PAX=("TOTAL_ELIGIBLE_PAX", "sum"),
    )
    .reset_index()
)

# Calculate overall percentage eligibility rate per tier
summary_df["ELIGIBILITY_RATE_%"] = (
    (summary_df["TOTAL_ELIGIBLE_PAX"] / summary_df["TOTAL_SEATS"]) * 100
).round(2)

# ==========================================
# 4. PRINT AND EXPORT SUMMARY
# ==========================================
print("\n--- Task 1: Operational Tier Summary Output ---")
print(summary_df.to_string(index=False))

# Export summary to CSV
output_csv = "task1_lounge_eligibility_summary.csv"
summary_df.to_csv(output_csv, index=False)
print(f"\n[Success] Output exported cleanly to '{output_csv}'.")