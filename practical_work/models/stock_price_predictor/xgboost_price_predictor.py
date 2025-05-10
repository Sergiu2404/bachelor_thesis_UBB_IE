# import yfinance as yf
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import math
#
#
# def get_stock_data(ticker):
#     df = yf.download(ticker, period="6mo")
#     return df[['Close']].dropna()
#
#
# def create_multistep_dataset(data, input_lags=10, output_days=10):
#     X, y = [], []
#     for i in range(len(data) - input_lags - output_days + 1):
#         X.append(data[i:i + input_lags])
#         y.append(data[i + input_lags:i + input_lags + output_days])
#     return np.array(X), np.array(y)
#
#
# def train_model(X, y):
#     base_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
#     model = MultiOutputRegressor(base_model)
#     model.fit(X, y)
#     return model
#
#
# def predict_next_days(model, last_sequence):
#     return model.predict([last_sequence])[0]
#
#
# def evaluate_model(y_true, y_pred):
#     metrics = {}
#
#     # Root Mean Squared Error
#     mse = mean_squared_error(y_true, y_pred)
#     metrics['RMSE'] = math.sqrt(mse)
#
#     # Mean Absolute Error
#     metrics['MAE'] = mean_absolute_error(y_true, y_pred)
#
#     # Mean Absolute Percentage Error
#     y_true_flat = y_true.flatten()
#     y_pred_flat = y_pred.flatten()
#     metrics['MAPE'] = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
#
#     # coefficient of determination
#     metrics['R2'] = r2_score(y_true, y_pred)
#
#     # dir accuracy - percentage of correct direction predictions
#     actual_directions = np.sign(np.diff(y_true.reshape(-1)))
#     predicted_directions = np.sign(np.diff(y_pred.reshape(-1)))
#     # remove zero direction predictions (no change)
#     valid_indices = np.where(actual_directions != 0)[0]
#     if len(valid_indices) > 0:
#         correct_directions = np.sum(actual_directions[valid_indices] == predicted_directions[valid_indices])
#         metrics['DA'] = correct_directions / len(valid_indices) * 100
#     else:
#         metrics['DA'] = np.nan
#
#     return metrics
#
#
# def main(ticker):
#     df = get_stock_data(ticker)
#     prices = df['Close'].values.reshape(-1)  # Ensure 1D array
#
#     input_lags = 10
#     output_days = 10
#
#     X, y = create_multistep_dataset(prices, input_lags, output_days)
#
#     # reshape from 3D to 2D for MultiOutputRegressor
#     X = X.reshape(X.shape[0], X.shape[1])
#
#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#
#     # Make sure y is 2D (samples, outputs)
#     if len(y.shape) == 3:
#         y = y.reshape(y.shape[0], y.shape[1])
#
#     model = train_model(X, y)
#
#     last_sequence = prices[-input_lags:]
#     predictions = predict_next_days(model, last_sequence)
#
#     print(f"\nNext {output_days}-day predictions for {ticker}:")
#     for i, price in enumerate(predictions, 1):
#         print(f"Day {i}: ${price:.2f}")
#
#     metrics = evaluate_model(y)
#
#
# if __name__ == "__main__":
#     main("AAPL")


import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


def get_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    return df[['Close']].dropna()


def create_multistep_dataset(data, input_lags=10, output_days=10):
    X, y = [], []
    for i in range(len(data) - input_lags - output_days + 1):
        X.append(data[i:i + input_lags])
        y.append(data[i + input_lags:i + input_lags + output_days])
    return np.array(X), np.array(y)


def train_model(X_train, y_train):
    base_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model


def predict_next_days(model, last_sequence):
    return model.predict([last_sequence])[0]


def evaluate_model(y_true, y_pred):
    metrics = {}

    # Root Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = math.sqrt(mse)

    # Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)

    # Mean Absolute Percentage Error
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics['MAPE'] = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100

    # eoefficient of determination
    metrics['R2'] = r2_score(y_true, y_pred)

    # Directional Accuracy: percentage of correct direction predictions
    actual_directions = np.sign(np.diff(y_true.reshape(-1)))
    predicted_directions = np.sign(np.diff(y_pred.reshape(-1)))

    # remove zero direction predictions (no change)
    valid_indices = np.where(actual_directions != 0)[0]
    if len(valid_indices) > 0:
        correct_directions = np.sum(actual_directions[valid_indices] == predicted_directions[valid_indices])
        metrics['DA'] = correct_directions / len(valid_indices) * 100
    else:
        metrics['DA'] = np.nan

    return metrics


def main(ticker, input_lags=10, output_days=10, test_size=0.2):
    df = get_stock_data(ticker)
    prices = df['Close'].values.reshape(-1)  # Ensure 1D array

    X, y = create_multistep_dataset(prices, input_lags, output_days)

    # Reshape X from 3D to 2D for MultiOutputRegressor
    X = X.reshape(X.shape[0], X.shape[1])

    # Make sure y is 2D (samples, outputs)
    if len(y.shape) == 3:
        y = y.reshape(y.shape[0], y.shape[1])

    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print(f"\nPerformance metrics for {ticker} using XGBoost:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    last_sequence = prices[-input_lags:]
    future_predictions = predict_next_days(model, last_sequence)

    print(f"\nNext {output_days}-day predictions for {ticker}:")
    for i, price in enumerate(future_predictions, 1):
        print(f"Day {i}: ${price:.2f}")

    return model, metrics


def compare_models(ticker, models_dict, input_lags=10, output_days=10, test_size=0.2):
    df = get_stock_data(ticker)
    prices = df['Close'].values.reshape(-1)

    X, y = create_multistep_dataset(prices, input_lags, output_days)
    X = X.reshape(X.shape[0], X.shape[1])
    if len(y.shape) == 3:
        y = y.reshape(y.shape[0], y.shape[1])

    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = {}

    for model_name, model_fn in models_dict.items():
        print(f"\nTraining {model_name}...")
        model = model_fn(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results[model_name] = metrics

        print(f"Performance metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(comparison_df)

    return comparison_df


if __name__ == "__main__":
    xgb_model, xgb_metrics = main("AAPL", input_lags=15, output_days=10)

    def train_xgb_default(X_train, y_train):
        base_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        return model


    def train_xgb_deep(X_train, y_train):
        base_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        return model


    def train_xgb_many_trees(X_train, y_train):
        base_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        return model

    models = {
        "XGBoost (Default)": train_xgb_default,
        "XGBoost (Deep Trees)": train_xgb_deep,
        "XGBoost (Many Trees)": train_xgb_many_trees
    }

    comparison_results = compare_models("AAPL", models, input_lags=15, output_days=10)