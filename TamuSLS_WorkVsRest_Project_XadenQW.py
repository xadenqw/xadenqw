import pandas as pd
from scipy.stats import pearsonr, zscore, ttest_ind
import numpy as np
import matplotlib.pyplot as plt

# Loading both datasets
df1 = pd.read_csv(r"C:\Users\xaden\Downloads\StatsProjectData\student_lifestyle_dataset.csv")
df2 = pd.read_csv(r"C:\Users\xaden\Downloads\StatsProjectData\CollegePlacement.csv")

# Clean up College_ID and numeric Student_ID (All data already cleaned in excel - no missing data/misspelt data)
df2["Student_ID"] = df2["College_ID"].str.extract(r"(\d+)$").astype(int)

# Merge datasets on Student_ID
merged = pd.merge(df1, df2, on="Student_ID", how="inner")

# Keep only Sleep and Employment columns
data = merged[["Sleep_Hours_Per_Day", "Placement"]].copy()

# Convert Placement to numeric (1 = employed, 0 = not)
data["Placement"] = data["Placement"].astype(str).str.lower().str.strip()
data["Employed"] = data["Placement"].map({
    "placed": 1, "yes": 1, "employed": 1,
    "not placed": 0, "no": 0, "unemployed": 0
})

# Drop rows where mapping failed
data = data.dropna(subset=["Employed"]).copy()
data["Employed"] = data["Employed"].astype(int)

sleep = data["Sleep_Hours_Per_Day"]
emp   = data["Employed"]

# Sleep Data (Mean/median/mode)
mean_sleep = data["Sleep_Hours_Per_Day"].mean()
median_sleep = data["Sleep_Hours_Per_Day"].median()
std_sleep = data["Sleep_Hours_Per_Day"].std()

print("\n🕒 Sleep Hours Summary Statistics:")
print(f"Mean Sleep Hours:   {mean_sleep:.2f} hours/day")
print(f"Median Sleep Hours: {median_sleep:.2f} hours/day")
print(f"Std. Deviation:     {std_sleep:.2f} hours/day")

# Getting quartile/IQR bounds
Q1, Q3 = np.percentile(sleep, [25, 75])
IQR    = Q3 - Q1
low_b  = Q1 - 1.5 * IQR
high_b = Q3 + 1.5 * IQR
clean  = data[(data["Sleep_Hours_Per_Day"] >= low_b) & (data["Sleep_Hours_Per_Day"] <= high_b)]
removed = len(data) - len(clean)
print("\n[Outlier Detection]")
print(f"IQR bounds: [{low_b:.2f}, {high_b:.2f}] | Removed {removed} rows as sleep outliers")
print(f"Cleaned dataset shape: {clean.shape}")

# Correlation analyses
pearson_r, pearson_p   = pearsonr(clean["Sleep_Hours_Per_Day"], clean["Employed"])
print("\n[Correlation]")
print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")

# T-test on sleep by employment group
sleep_emp1 = clean.loc[clean["Employed"] == 1, "Sleep_Hours_Per_Day"]
sleep_emp0 = clean.loc[clean["Employed"] == 0, "Sleep_Hours_Per_Day"]
t_stat, p_val = ttest_ind(sleep_emp1, sleep_emp0, equal_var=False, nan_policy="omit")

# Print T-test
print("\n[Hypothesis Testing: difference in mean sleep (Employed vs. Unemployed)]")
print(f"Means -> Employed: {sleep_emp1.mean():.2f} h, Not Employed: {sleep_emp0.mean():.2f} h")
print(f"Welch t-test: t = {t_stat:.3f}, p = {p_val:.4f}  {'(significant)' if p_val < 0.05 else '(not significant)'}")

# Print Z-Score
clean["Sleep_Z_score"] = zscore(clean["Sleep_Hours_Per_Day"])
print("\n[Z-test results]")
print("Sleep_Hours_Per_Day converted to z-scores -> column 'Sleep_Z_score' (mean≈0, std≈1)")
print(clean[["Sleep_Hours_Per_Day", "Sleep_Z_score"]].head(3))

# Final Analysis intro
print("\n---------------")
print("Final Analysis")
print("----------------")

# Final Analysis Statement
mean_employed = clean.loc[clean["Employed"] == 1, "Sleep_Hours_Per_Day"].mean()
mean_unemployed = clean.loc[clean["Employed"] == 0, "Sleep_Hours_Per_Day"].mean()

if mean_employed > mean_unemployed:
    print(f"\nUsing this data, we understand that employed students sleep more on average "
          f"({mean_employed:.2f} hrs/day) than students who are not employed ({mean_unemployed:.2f} hrs/day).\nTherefore our hypothesis is incorrect.")
elif mean_employed < mean_unemployed:
    print(f"\nUsing this data, we understand that employed students sleep less on average "
          f"({mean_employed:.2f} hrs/day) than students who are not employed ({mean_unemployed:.2f} hrs/day).\nTherefor our hypothesis is correct.")
else:
    print(f"\nUsing this data, we understand that employed and unemployed students sleep about the same on average "
          f"({mean_employed:.2f} hrs/day).")

# Compute linear regression (y = m*x + b) and print
x = clean["Sleep_Hours_Per_Day"]
y = clean["Employed"]

m, b = np.polyfit(x, y, 1)
y_pred = m * x + b
r_squared = np.corrcoef(x, y)[0, 1] ** 2

print(f"\n📈 Linear Approximation:")
print(f"Equation: y = {m:.4f}x + {b:.4f}")
print(f"R² (coefficient of determination): {r_squared:.4f}")

# Print results of the Pearson equation
print("\n📊 Correlation Between Sleep Hours and Employment Status:")
print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")

# Determine if there's a significant relationship based on 0.05
if pearson_p < 0.05:
    print("✅ Statistically significant relationship (p < 0.05)")
else:
    print("❌ No statistically significant relationship (p ≥ 0.05)")

# Histogram of Sleep Hours Per Night (using cleaned data)
plt.figure(figsize=(10, 6))
plt.hist(clean["Sleep_Hours_Per_Day"], bins=20, alpha=0.75, edgecolor="black")

plt.xlabel("Sleep Hours Per Night")
plt.ylabel("Frequency")
plt.title("Histogram of Sleep Hours Per Night (Cleaned Data)")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Boxplot showing the distribution of hours slept
plt.figure(figsize=(10, 6))
plt.boxplot(clean["Sleep_Hours_Per_Day"], vert=False, patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="red", linewidth=2))

plt.xlabel("Sleep Hours Per Night")
plt.title("Horizontal Boxplot of Sleep Hours Per Night (Cleaned Data)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Bar Chart of Employed vs. Unemployed
counts = clean["Employed"].value_counts().sort_index()

ordered_counts = [counts[1], counts[0]]
labels = ["Employed", "Unemployed"]

plt.figure(figsize=(10, 6))

plt.bar(labels, ordered_counts, color=["lightblue", "blue"], edgecolor="black")

plt.ylabel("Number of Students")
plt.xlabel("Employment Status")
plt.title("Number of College Students: Employed vs Unemployed")

plt.grid(axis="y", linestyle="--", alpha=0.5)

for i, v in enumerate(ordered_counts):
    plt.text(i, v + 50, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Side by side box plots for employed vs. unemployed
plt.figure(figsize=(10, 6))

sleep_emp = clean.loc[clean["Employed"] == 1, "Sleep_Hours_Per_Day"]
sleep_unemp = clean.loc[clean["Employed"] == 0, "Sleep_Hours_Per_Day"]

box = plt.boxplot(
    [sleep_emp, sleep_unemp],
    vert=False,
    labels=["Employed", "Unemployed"],
    patch_artist=True,
    medianprops=dict(color="red", linewidth=2)
)

colors = ["lightblue", "blue"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel("Sleep Hours Per Night")
plt.ylabel("Employment Status")
plt.title("Horizontal Boxplot of Sleep Hours by Employment Status")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()