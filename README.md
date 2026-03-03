# Corn Futures Price Prediction

Analysis of machine learning models for predicting U.S. corn futures prices using external market and environmental factors.\
[Full Paper](https://github.com/ivanfye/corn-futures/blob/main/Analysis%20of%20Corn%20Futures.pdf)

## Overview

This project investigates which external factors most significantly influence U.S. corn futures prices and compares the predictive performance of four distinct machine learning architectures. Rather than relying on historical corn price data, the models are trained exclusively on related market indicators and weather data to identify meaningful feature relationships.

## Key Findings

- **Related U.S. crop prices** (oats, rice, wheat) show the strongest correlation with corn futures prices
- **Weather data** has minimal predictive value for corn futures, confirming prior research
- **International corn prices** provide moderate predictive signals
- **Model-specific strengths:** CNN and RNN outperform the feed-forward baseline on crop and international data; the Time Series Transformer shows competitive performance on international inputs

## Dataset

| Source | Description |
|--------|-------------|
| **U.S. Corn Futures** | Primary target variable from the Chicago Board of Trade |
| **Related Crop Futures** | Oats, rice, and wheat prices (CBOT) |
| **International Corn Futures** | Dalian (China) and São Paulo (Brazil) exchanges |
| **Weather Data** | Iowa and Illinois — primary U.S. corn-growing regions |

Dates with missing values across any category are excluded. CNN models use 18 sequential dates (~30 days) as input windows. RNN data is normalized with `MinMaxScaler` and split 85/15 train/test.

## Models

1. Feed-Forward Neural Network
2. Convolutional Neural Network (CNN)
3. Recurrent Neural Network (RNN)
4. Time Series Transformer

## Limitations & Future Work

- **Economic shocks:** Events like COVID-19 introduce anomalies that reduce model accuracy
- **Weather signals:** Despite low direct correlation, indirect effects through crop supply chains may warrant further study
- **Transformer complexity:** Higher training loss suggests the dataset may be too small for the model's capacity

Next steps:
- Add lagged features and rolling averages
- Explore ensemble methods combining multiple model types
- Extend to longer time horizons and additional market regimes
- Add prediction intervals / uncertainty quantification
