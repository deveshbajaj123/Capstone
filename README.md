Tea production in Darjeeling-like regions is shaped by climate, soil conditions, vegetation health, and management practices. However, long-term yield data is scarce and fragmented.

To address this, the project:

Builds a multi-decadal (1940–2025) dataset using real measurements + statistically grounded synthetic generation

Uses remote sensing (NDVI, RVI) to capture vegetation dynamics

Trains an XGBoost regression model to predict annual tea yield

Incorporates ERW as a predictive feature

Conducts detailed analysis using feature importance, correlations, and time-series behavior

Key Components
1. Multi-Decadal Dataset Construction

A unified dataset was created by combining:

Real data sources

IMD rainfall (gridded datasets, CRIS)

Daily temperature (WeatherAndClimate.eu)

Computed NDVI & RVI via Google Earth / remote sensing workflows

Fertiliser & yield records from Kamala Tea Gardens (2017–2024)

Cropping intensity from Siliguri tehsil datasets

Synthetic extensions

To cover 1940–2016:

Gaussian processes for NDVI/RVI extrapolation

Trend-guided inference for fertilizer practices & cropping intensity

Yield reconstruction using:

Mean + SD anchoring

Historical agronomic phases

Controlled smoothing + variance modeling

This produced the first robust long-term dataset integrating climate × vegetation × management × ERW.

Machine Learning Pipeline

Model: XGBoost Regressor

Pipeline steps:

Data cleaning & encoding

Feature-target separation

80:20 train-test split

Model training with tuned hyperparameters

Prediction, regression metrics, and time-series evaluation

Feature importance analysis

Results
With NDVI + RVI

R² ≈ 0.94

RMSE ~250–300 kg/ha

Accurately captures long-run yield levels and trends

Without NDVI + RVI

R² ≈ 0.63

RMSE ~358 kg/ha

Shows strong dependence on remote-sensing features

Interpretability Highlights

NDVI = most important feature

Followed by cropping intensity, RVI, ERW indicator, and fertilizer quantity

Climate variables have modest direct impact

Time-series plots show correct trend modeling with expected smoothing of extremes

Enhanced Rock Weathering (ERW) Integration

ERW was introduced only in 2024–2025.
The project:

Encodes ERW as a binary feature

Examines model behavior in ERW years

Notes that causal effects cannot be inferred due to limited treated years

Demonstrates how ERW can be incorporated into predictive frameworks for future analysis

Visual & Analytical Outputs

The project generates:

Predicted vs actual yield scatterplots

Time-series plots across 85+ years

Feature importance rankings

Correlation heatmaps

NDVI/RVI yearly trajectories

Decadal yield summaries

These enable scientific interpretation of vegetation–climate–yield relationships.

Tech Stack

Python

Pandas, NumPy

Scikit-Learn, XGBoost

Matplotlib, Seaborn

Remote Sensing (NDVI, RVI workflows)

Future Scope

This framework is designed to scale with new data.
Next steps include:

Integrating multi-estate datasets

Expanding ERW data for causal modeling

Deploying forecasting pipelines for real-time estate support

Scenario modeling under climate-change projections

As richer real-world satellite and estate-level data becomes available, this system can evolve into a full agricultural forecasting engine.
