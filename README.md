# Tea Yield Prediction Using Climate, Remote Sensing and ERW

This project studies tea yield in Darjeeling-like regions using climate, vegetation, management and ERW-related features.

The goal is to build a long-term dataset and use machine learning to predict annual tea yield and driving factors of the same.

## Dataset

The dataset covers 1940-2025 and combines real data with statistically generated historical estimates.

Real data includes:

- IMD rainfall data
- Daily temperature data
- NDVI and RVI
- Fertilizer and yield records from Kamala Tea Gardens
- Cropping intensity data

For earlier years where complete data was not available, values were estimated using:

- Gaussian process extrapolation
- Trend-based estimation
- Historical averages and variation
- Controlled smoothing

## Model

The main model used is XGBoost Regressor.

The pipeline includes:

- Data cleaning
- Feature selection
- 80:20 train-test split
- Model training
- Prediction
- Evaluation
- Feature importance analysis

## Results

With NDVI and RVI:

- R2: approximately 0.94
- RMSE: approximately 250-300 kg/ha

Without NDVI and RVI:

- R2: approximately 0.63
- RMSE: approximately 358 kg/ha

This shows that remote sensing features add significant predictive value.

## Feature Importance

The most important features were:

- NDVI
- Cropping intensity
- RVI
- ERW indicator
- Fertilizer quantity

Climate variables had a smaller direct effect in the model.

## ERW

Enhanced Rock Weathering was added as a binary feature for 2024-2025.

Since there are only a few years of ERW data, the project does not make causal claims about its effect on yield.

The current goal is to include ERW in the prediction framework and study its effect as more data becomes available.

## Visualisations

The project includes:

- Predicted vs actual yield plots
- Yield time-series plots
- Feature importance plots
- Correlation heatmaps
- NDVI and RVI trends
- Decadal yield summaries


## Future Work

- Add data from more tea estates
- Add more ERW data
- Improve satellite data coverage
- Build forecasting models
- Test climate change scenarios
