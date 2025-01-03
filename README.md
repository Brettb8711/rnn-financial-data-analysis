
# RNN Financial Data Analysis
## Project Overview
This project explores the use of Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) models, to analyze and predict financial data. The goal is to assess the effectiveness of RNNs in capturing trends and patterns in stock prices and index movements.

By leveraging historical data of major companies and indices (e.g., SPY, AAPL, MSFT), the project aims to:

- Predict future stock prices based on historical data.
- Evaluate the performance of RNN models against benchmark methods.
- Gain insights into preprocessing techniques and model tuning for financial data.


## Features
- Data collection using yfinance.
- Data preprocessing, including normalization and sequence creation.
- Implementation of LSTM models for time series forecasting.
- Visualizing model predictions versus actual values.


## Project Structure
```
rnn-financial-data-analysis/
│
├── data/
│   ├── X_train.npy        # Training features
│   ├── X_test.npy         # Testing features
│   ├── y_train.npy        # Training labels
│   ├── y_test.npy         # Testing labels
│   ├── scaler.pkl         # Scaler object for normalization
│
├── notebooks/
│   ├── data_preprocessing.ipynb   # Data collection and preprocessing steps
│
├── models/
│   ├── rnn_model.h5       # Saved trained model
│
├── src/
│   ├── train_model.py     # Script for training and evaluating the model
│
├── README.md              # Project documentation
```
## Progress

### Milestones Achieved
- ✅ Data collection from yfinance for SPY, AAPL, MSFT, GOOGL, and AMZN.
- ✅ Preprocessed the data: handled missing values, normalized, and created sequences.
- ✅ Split the data into training and testing sets and saved preprocessed data.
- ✅ Implemented a baseline LSTM model.
- ✅ Evaluated the model's performance on test data.

### Current Focus
- Fine-tuning the LSTM architecture and hyperparameters.
- Exploring additional tickers and multi-ticker predictions.
- Future Goals
- Incorporate multi-variable models (e.g., using Volume, Open, High, Low features).
- Experiment with other time-series models (GRUs, Transformers).
- Deploy the model as an API for real-time predictions.

## How to Use
### Setup
Clone this repository:

git clone https://github.com/yourusername/rnn-financial-data-analysis.git
cd rnn-financial-data-analysis

Install dependencies:

pip install -r requirements.txt
Run the preprocessing notebook:

notebooks/data_preprocessing.ipynb
This will download, clean, and preprocess the data.

Train the model:

python src/train_model.py


### Requirements
Python 3.8+
Libraries: yfinance, numpy, pandas, scikit-learn, tensorflow, matplotlib

## Results
Add visuals like loss curves, actual vs. predicted stock prices, or key performance metrics (e.g., MSE, MAE) here as you progress.

## Acknowledgments
- yfinance for financial data collection.
- TensorFlow for machine learning tools.
- Online tutorials and the open-source community for inspiration.
- ChatGPT for programming assitance and organizing the README
