📊 Stock Price Anomaly Detection and Forecasting using Machine Learning
This project focuses on detecting stock price anomalies and predicting future closing prices using a hybrid approach combining technical indicators, rule-based anomaly detection, unsupervised machine learning (Isolation Forest), and deep learning (LSTM). The project uses the Yahoo Finance 5-year stock dataset for major companies like AAPL, MSFT, GOOGL, AMZN, and NVDA.

🔍 Objectives
Detect anomalous stock price behavior using:

Rule-based conditions on technical indicators.

Isolation Forest for unsupervised anomaly detection.

Predict future stock closing prices using a Long Short-Term Memory (LSTM) neural network.

Visualize trends, anomalies, and prediction performance.

📁 Dataset
Source: Massive Yahoo Finance Dataset (Kaggle)

Dataset: stock_details_5_years.csv

Fields: Date, Open, High, Low, Close, Volume, Company

🧪 Methodology
1. Data Preprocessing
Parsed Date column as datetime and removed timezone.

Filtered for five major companies: AAPL, MSFT, GOOGL, AMZN, NVDA.

Handled missing values by dropping incomplete rows.

Ensured unique dates per company and sorted chronologically.

2. Technical Indicators Calculation
For each company:

SMA (Simple Moving Average) - 20 days

EMA (Exponential Moving Average) - 20 days

RSI (Relative Strength Index) - 14 days

Bollinger Bands (Upper & Lower)

These features are used for both anomaly detection and as input features for ML models.

3. Rule-Based Anomaly Detection
Applied traditional trading rule checks:

RSI > 70 (overbought) or RSI < 30 (oversold)

Price outside Bollinger Bands

Close price deviates >5% from SMA

Marked any condition match as an anomaly.

4. Isolation Forest (Unsupervised Anomaly Detection)
Scaled features using StandardScaler.

Used the following features:

Close, SMA_20, EMA_20, RSI_14, BB_upper, BB_lower

Trained Isolation Forest (contamination=0.1) for anomaly detection.

Stored anomalies for each company.

Saved trained model using pickle.

5. LSTM Model for Price Prediction
Normalized Close price using MinMaxScaler.

Generated sequences of 20 previous days to predict the next day's price.

Built LSTM architecture:

2 LSTM layers (50 units each)

1 Dense output layer

Used callbacks for:

Early stopping

Model checkpointing

TensorBoard logging

Trained the model on 80% of data and validated on 20%.

Saved final trained model.

6. Evaluation Metrics
RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) calculated for LSTM predictions.

Isolation Forest evaluated by counting detected anomalies in test and full datasets.

📊 Visualizations
Stock Price with Anomalies (Isolation Forest)

LSTM Predicted vs Actual Prices

Training vs Validation Loss (LSTM)

Prediction Residual Plot

Residual Distribution (Histogram)

These visualizations provide insight into model performance and anomaly distribution.

📂 Project Structure
├── data/
│   └── stock_details_5_years.csv
├── notebooks/
│   └── stock_anomaly_lstm.ipynb
├── models/
│   ├── iso_forest_model.pkl
│   └── lstm_AAPL_final.keras
├── logs/
│   └── tensorboard_logs/
├── outputs/
│   ├── plots/
│   └── evaluation_metrics.txt
└── README.md

✅ Results (Example: AAPL)
LSTM RMSE: 4.10

LSTM MAE: 3.02

IF Anomalies in Test Set: 39

IF Anomalies in Full Dataset: 59

💡 Future Improvements
Incorporate sentiment analysis (news headlines or tweets).

Use Transformer-based models for sequence prediction.

Apply unsupervised deep anomaly detectors like Autoencoders or GANs.

Evaluate financial impact of anomalies (e.g., profit/loss from trading decisions).

📌 Requirements
Python 3.8+

TensorFlow 2.x

scikit-learn

matplotlib, seaborn

pandas, numpy

🚀 How to Run
Clone the repo:

git clone https://github.com/your-username/stock-anomaly-lstm.git
cd stock-anomaly-lstm



✍️ Author
Haseeb Ur Rehman

