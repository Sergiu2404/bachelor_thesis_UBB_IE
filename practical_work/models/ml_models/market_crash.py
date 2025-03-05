import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # To save/load the trained model


def fetch_and_save_data(filename='sp500_data.csv', start_date='1970-01-01'):
    """Fetch S&P 500 data once and save it locally."""
    sp500 = yf.download('^GSPC', start=start_date)
    sp500.to_csv(filename)
    return sp500


def load_data(filename='sp500_data.csv'):
    """Load historical data from file to avoid repeated API calls."""
    return pd.read_csv(filename, index_col='Date', parse_dates=True)


def create_features(df):
    """Generate predictive financial indicators."""
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_21d'] = df['Close'].pct_change(21)
    df['volatility_21d'] = df['return_1d'].rolling(21).std()
    df['ma_50d'] = df['Close'].rolling(50).mean()
    df['ma_200d'] = df['Close'].rolling(200).mean()
    df['ma_ratio'] = df['ma_50d'] / df['ma_200d']
    df['bb_middle'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['rsi_14d'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() /
                                       -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['crash_label'] = (df['Close'].pct_change(21).shift(-21) < -0.2).astype(int)
    return df.dropna()


def train_and_save_model(X, y, model_filename='crash_model.pkl', scaler_filename='scaler.pkl'):
    """Train a RandomForest model and save it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("Model Performance:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, model_filename)
    joblib.dump(scaler, scaler_filename)
    return clf, scaler


def predict_next_crash(model_filename='crash_model.pkl', scaler_filename='scaler.pkl', data_filename='sp500_data.csv'):
    """Load the trained model and predict future crash probability."""
    clf = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    df = load_data(data_filename)
    df = create_features(df)
    X_latest = df.drop(['crash_label', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1).iloc[-1:]
    X_latest_scaled = scaler.transform(X_latest)
    crash_prob = clf.predict_proba(X_latest_scaled)[0, 1]
    print(f"Predicted Crash Probability: {crash_prob:.2%}")
    return crash_prob


if __name__ == "__main__":
    try:
        data = load_data()
    except FileNotFoundError:
        data = fetch_and_save_data()

    print("Training model...")
    data = create_features(data)
    X = data.drop(['crash_label', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    y = data['crash_label']
    train_and_save_model(X, y)

    print("Predicting next crash...")
    predict_next_crash()
