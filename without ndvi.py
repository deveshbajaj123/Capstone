import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# LOAD + CLEAN DATA
# -------------------------------
df = pd.read_excel("Capstone - dataset.xlsx")

# Remove unnamed junk columns
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c or "Column1" in c])

# Encode fertiliser type
df["Fertiliser/ Manure type"] = df["Fertiliser/ Manure type"].astype("category").cat.codes

# Encode ERW
df["ERW Indicator (Yes/No)"] = df["ERW Indicator (Yes/No)"].map({"Yes":1,"No":0})

# Convert everything possible to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop bad rows
df = df.dropna()

# -------------------------------
# REMOVE NDVI
# -------------------------------
if "NDVI" in df.columns:
    df = df.drop(columns=["NDVI"])
else:
    print("NDVI column not found, proceeding anyway.")

if "RVI" in df.columns:
    df = df.drop(columns=["RVI"])
else:
    print("NDVI column not found, proceeding anyway.")

# -------------------------------
# SPLIT FEATURES / TARGET
# -------------------------------
X = df.drop(columns=["Crop Yield (kg)"])
y = df["Crop Yield (kg)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# XGBOOST MODEL
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
# PREDICT + METRICS
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n========== MODEL PERFORMANCE (NDVI REMOVED) ==========")
print("R² Score (without NDVI):", r2)
print("RMSE:", rmse)

# -------------------------------
# SCATTER: ACTUAL VS PREDICTED
# -------------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield (NDVI Removed)")
plt.grid(True)
plt.show()

# -------------------------------
# CONTINUOUS TIME SERIES PLOT
# -------------------------------
test_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
test_results = test_results.sort_index()

plt.figure(figsize=(12,6))
plt.plot(test_results.index, test_results["Actual"], label="Actual Yield", linewidth=2)
plt.plot(test_results.index, test_results["Predicted"], label="Predicted Yield", linewidth=2)

plt.xlabel("Time Index")
plt.ylabel("Crop Yield")
plt.title("Continuous Actual vs Predicted Yield (NDVI Removed)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance (NDVI Removed)")
plt.tight_layout()
plt.show()

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="viridis")
plt.title("Correlation Heatmap (NDVI Removed)")
plt.show()
