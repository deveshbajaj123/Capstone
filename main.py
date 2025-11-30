import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# 1. LOAD + CLEAN DATA
# -------------------------------
df = pd.read_excel("Capstone - dataset (1).xlsx")

# Drop unnamed junk columns
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c or "Column1" in c])

# Encode Fertiliser/Manure type
df["Fertiliser/ Manure type"] = df["Fertiliser/ Manure type"].astype("category").cat.codes

# Encode ERW indicator
df["ERW Indicator (Yes/No)"] = df["ERW Indicator (Yes/No)"].map({"Yes": 1, "No": 0})

# Convert all numeric-like columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows that became NaN after conversion
df = df.dropna()

# -------------------------------
# 2. TRAIN-TEST SPLIT
# -------------------------------
X = df.drop(columns=["Crop Yield (kg)"])
y = df["Crop Yield (kg)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. XGBOOST MODEL
# -------------------------------
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 4. PREDICTIONS & METRICS
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("RMSE:", rmse)

# -------------------------------
# 5. ACTUAL vs PREDICTED SCATTER PLOT
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("XGBoost: Actual vs Predicted Yield (Scatter Plot)")
plt.grid(True)
plt.show()


# -----------------------------------------------------
# 6. CONTINUOUS TIME-SERIES PLOT (Requested)
# -----------------------------------------------------

# Restore chronological order of the test set
# (train_test_split shuffles rows, so we re-sort by dataframe index)
test_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
test_results = test_results.sort_index()

plt.figure(figsize=(12,6))
plt.plot(test_results.index, test_results["Actual"], label="Actual Yield", linewidth=2)
plt.plot(test_results.index, test_results["Predicted"], label="Predicted Yield", linewidth=2)

plt.xlabel("Time (Index Order)")
plt.ylabel("Crop Yield (kg)")
plt.title("Continuous Actual vs Predicted Crop Yield Over Time")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------
# 7. FEATURE IMPORTANCE
# -------------------------------
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.xlabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------------
# 8. CORRELATION HEATMAP
# -------------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="viridis")
plt.title("Feature Correlation Heatmap")
plt.show()
