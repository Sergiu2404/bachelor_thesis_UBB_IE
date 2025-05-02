# import numpy as np
# import pandas as pd
# import yfinance as yf
# import datetime
# from dateutil import tz
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import tensorflow as tf
# import requests
# # from keras.layers import Bidirectional, Dense, Dropout, LSTM
# # from keras.callbacks import EarlyStopping, ModelCheckpoint
# import requests
#
#
# print("All imports successful.")
#
#
# import warnings
#
# warnings.filterwarnings('ignore')
#
# import os
#
# model_path_dir = r"E:\saved_models\LSTM_price_prediction_model"
# model_path_file = os.path.join(model_path_dir, "LSTM_price_prediction_model.keras")
#
# model_exists = os.path.exists(model_path_file) and os.path.getsize(model_path_file) > 0
#
# if model_exists:
#     print("Model found. Loading existing model...")
#     model = keras.models.load_model(model_path_file)
# else:
#     print("No saved model found. Training new model...")
#
#
# try:
#     session = requests.Session(impersonate="chrome")
#     response = session.get("https://www.google.com")
#     print("Success:", response.status_code)
# except Exception as e:
#     print("Failed impersonation:", e)
#
# session = requests.Session(impersonate="chrome")
# ticker = yf.Ticker("^GSPC", session=session)
# data = ticker.history(period="5y")
# data = data.reset_index()
# print(data.head())
#
# data['Date'] = pd.to_datetime(data['Date'])
# stock_close = data.filter(['Close'])
# dataset = stock_close.values  # Convert to numpy array
#
# train_data_len = int(np.ceil(len(dataset) * 0.95))
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(dataset)
# training_data = scaled_data[:train_data_len]
#
#
# sequence_length = 60
# X_train, y_train = [], []
# for i in range(sequence_length, len(training_data)):
#     X_train.append(training_data[i - sequence_length:i, 0])
#     y_train.append(training_data[i, 0])
#
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#
# if not model_exists:
#     model = keras.Sequential()
#
#     model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
#     model.add(Dropout(0.2))
#
#     model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
#     model.add(Dropout(0.2))
#
#     model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(units=50, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=25, activation='relu'))
#     model.add(Dense(units=1))
#
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])
#
#     # Train with EarlyStopping and validation
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     training = model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=32,
#         validation_split=0.1,
#         callbacks=[early_stop],
#         verbose=1
#     )
#
#     os.makedirs(model_path_dir, exist_ok=True)
#     model.save(model_path_file)
#     print(f"Model saved to: {model_path_file}")
#
# test_data = scaled_data[train_data_len - sequence_length:]
# X_test, y_test = [], dataset[train_data_len:]
#
# for i in range(sequence_length, len(test_data)):
#     X_test.append(test_data[i - sequence_length:i, 0])
#
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
# predictions = model.predict(X_test)
# predictions = scaler.inverse_transform(predictions)
#
# rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
# print(f"Root Mean Squared Error (RMSE): {rmse}")
#
# last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
#
# future_predictions = []
# current_batch = last_sequence.copy()
#
# for _ in range(10):
#     current_pred = model.predict(current_batch)[0]
#     future_predictions.append(current_pred[0])
#     current_batch = np.append(current_batch[:, 1:, :],
#                               [[current_pred]],
#                               axis=1)
#
# future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
#
# last_date = data['Date'].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')
#
# future_df = pd.DataFrame({
#     'Date': future_dates,
#     'Predicted_Close': future_prices.flatten()
# })
#
# print("\nPredictions for the next 10 business days:")
# for i, (date, price) in enumerate(zip(future_dates, future_prices.flatten())):
#     print(f"Day {i + 1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
#
# test = data[train_data_len:].copy()
# test['Predictions'] = np.nan
#
# min_len = min(len(test), len(predictions))
# test.iloc[:min_len, test.columns.get_loc('Predictions')] = predictions.flatten()[:min_len]
#
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# mae = mean_absolute_error(test['Close'][:min_len], test['Predictions'][:min_len])
# mse = mean_squared_error(test['Close'][:min_len], test['Predictions'][:min_len])
# rmse = np.sqrt(mse)
# r2 = r2_score(test['Close'][:min_len], test['Predictions'][:min_len])
#
# print("\nModel Performance Metrics:")
# print(f"Mean Absolute Error (MAE): ${mae:.2f}")
# print(f"Mean Squared Error (MSE): ${mse:.2f}")
# print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
# print(f"R-squared (R²): {r2:.4f}")
#
# accuracy = 100 - (mae / test['Close'][:min_len].mean() * 100)
# print(f"Approximate Prediction Accuracy: {accuracy:.2f}%")

# future_df.to_csv('future_stock_predictions.csv', index=False)
# test[['Date', 'Close', 'Predictions']].to_csv('test_predictions.csv', index=False)
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))












# # Method 1: Standard version check
# import tensorflow as tf
#
# try:
#     print("TensorFlow version (tf.__version__):", tf.__version__)
# except AttributeError:
#     print("AttributeError: tf.__version__ not found")
#
# # Method 2: Alternative ways to check TensorFlow version
# try:
#     print("\nAlternative version checks:")
#     print("From VERSION:", getattr(tf, 'VERSION', 'Not available'))
#     print("From version:", getattr(tf, 'version', 'Not available'))
#     if hasattr(tf, 'version'):
#         print("From tf.version.VERSION:", getattr(tf.version, 'VERSION', 'Not available'))
# except Exception as e:
#     print(f"Error checking alternative versions: {e}")
#
# # Method 3: Check TensorFlow installation path
# print("\nTensorFlow installation path:")
# print(tf.__file__)
#
# # Method 4: Check if TensorFlow is properly installed with pip
# import subprocess
# import sys
#
# print("\nChecking pip installation:")
# try:
#     pip_result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'tensorflow'],
#                                 capture_output=True, text=True)
#     print(pip_result.stdout)
# except Exception as e:
#     print(f"Error checking pip: {e}")
#
# # Check for potential module conflicts
# print("\nChecking for potential module conflicts:")
# try:
#     import importlib
#
#     spec = importlib.util.find_spec("tensorflow")
#     print(f"TensorFlow spec: {spec}")
#
#     # Check if there's a name conflict
#     import sys
#
#     conflicting_modules = [name for name in sys.modules
#                            if name.startswith('tensorflow') and name != 'tensorflow']
#     print(f"Potential conflicting modules: {conflicting_modules}")
# except Exception as e:
#     print(f"Error checking conflicts: {e}")
#
