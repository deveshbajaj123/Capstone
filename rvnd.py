import numpy as np
import pandas as pd

# -----------------------------
# SETTINGS
# -----------------------------
start_year = 1940
end_year   = 2025
years = np.arange(start_year, end_year + 1)

np.random.seed(42)  # for reproducibility

# -----------------------------
# 1. NDVI BASELINE TREND
# -----------------------------
def ndvi_baseline(year):
    # Piecewise trend (roughly realistic for Darjeeling tea)
    if year <= 1960:
        # 1940–1960: 0.58 -> 0.64
        return 0.58 + (year - 1940) * (0.64 - 0.58) / (1960 - 1940)
    elif year <= 1985:
        # 1960–1985: 0.64 -> 0.70
        return 0.64 + (year - 1960) * (0.70 - 0.64) / (1985 - 1960)
    elif year <= 2005:
        # 1985–2005: 0.70 -> 0.69 (slight decline)
        return 0.70 + (year - 1985) * (0.69 - 0.70) / (2005 - 1985)
    else:
        # 2005–2025: 0.69 -> 0.66 (climate stress)
        return 0.69 + (year - 2005) * (0.66 - 0.69) / (2025 - 2005)

baseline_ndvi = np.array([ndvi_baseline(y) for y in years])

# -----------------------------
# 2. AR(1) NOISE AROUND TREND
# -----------------------------
rho = 0.7      # temporal persistence
sigma = 0.02   # base noise scale

ndvi = np.zeros_like(baseline_ndvi)
ndvi[0] = baseline_ndvi[0] + np.random.normal(0, sigma)

for t in range(1, len(years)):
    # AR(1) around the *trend*
    prev_dev = ndvi[t-1] - baseline_ndvi[t-1]
    eps = np.random.normal(0, sigma * np.sqrt(1 - rho**2))
    ndvi[t] = baseline_ndvi[t] + rho * prev_dev + eps

# Clip NDVI to realistic range
ndvi = np.clip(ndvi, 0.45, 0.85)

# -----------------------------
# 3. APPLY ERW SHOCKS (EXTRA DIPS)
# # -----------------------------
# erw_years = {2024: -0.05, 2025: -0.03}  # extra negative shocks
#
# for y, shock in erw_years.items():
#     idx = np.where(years == y)[0]
#     if len(idx) > 0:
#         ndvi[idx[0]] = np.clip(ndvi[idx[0]] + shock, 0.45, 0.85)

# -----------------------------
# 4. DERIVE RVI FROM NDVI WITH NOISE
# -----------------------------
# Simple linear-ish relation + noise
# Centered around NDVI ~ 0.65 → RVI ~ 0.80
a = 0.80   # intercept
b = 0.4    # slope

rvi_noise_sigma = 0.01
rvi = a + b * (ndvi - 0.65) + np.random.normal(0, rvi_noise_sigma, size=len(ndvi))

# Clip RVI to realistic range
rvi = np.clip(rvi, 0.65, 0.90)

# -----------------------------
# 5. BUILD DATAFRAME
# -----------------------------
df_indices = pd.DataFrame({
    "Year": years,
    "NDVI": ndvi,
    "RVI": rvi
})

print(df_indices.head(15))
print(df_indices.tail(10))

# -----------------------------
# 6. OPTIONALLY SAVE TO CSV
# -----------------------------
df_indices.to_csv("realistic_ndvi_rvi_1940_2025.csv", index=False)
print("Saved: realistic_ndvi_rvi_1940_2025.csv")
