# Corn Futures Price Prediction

Analysis of machine learning models for predicting U.S. corn futures prices using external market and environmental factors. 

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

### 1. Feed-Forward Neural Network
Baseline architecture with three hidden layers (70 → 50 → 30 units), Leaky ReLU activation, and dropout (0.1).
- **Loss:** Mean Squared Error
- **Learning rate:** 0.0002 (crop/international), 0.0001 (weather)
- **Batch size / Epochs:** 64 / 250

### 2. Convolutional Neural Network (CNN)
Adapted for sequential data using 1D convolutions to detect local temporal patterns.
- **Architecture:** Two conv blocks (kernel sizes 4 and 2, channels 2 and 4), average pooling, feed-forward head
- **Learning rate / Epochs:** 0.001 / 100 with early stopping, weight decay 0.0001
- **Ensemble:** 10 CNNs trained per input category

### 3. Recurrent Neural Network (RNN)
Captures temporal dependencies across sequential price data.
- **Architecture:** RNN layer → tanh activation → dropout → fully connected
- **Loss:** Mean Squared Error
- **Learning rate / Epochs:** 0.0001 / 200, batch size 64

### 4. Time Series Transformer
Leverages self-attention to dynamically weight the importance of past observations.
- **Architecture:** 3 encoder layers, learned positional encodings, multi-head attention (`nhead` = number of parameters)

## Results

| Model | Crop Data (MSE) | International (MSE) | Weather (MSE) |
|-------|:-----------:|:------------------:|:-----------:|
| Feed-Forward | 4,371.09 | 14,012.17 | 11,373.40 |
| CNN | 3,591.67 | — | — |
| RNN | 3,810.66 | — | 33,845.21 |
| Time Series Transformer | 6,215.67 | 5,097.51 | — |

*Lower is better. Related U.S. crop prices consistently yield the lowest test loss across models.*


## Limitations & Future Work

- **Economic shocks:** Events like COVID-19 introduce anomalies that reduce model accuracy
- **Weather signals:** Despite low direct correlation, indirect effects through crop supply chains may warrant further study
- **Transformer complexity:** Higher training loss suggests the dataset may be too small for the model's capacity

Recommended next steps:
- Add lagged features and rolling averages
- Explore ensemble methods combining multiple model types
- Extend to longer time horizons and additional market regimes
- Add prediction intervals / uncertainty quantification
