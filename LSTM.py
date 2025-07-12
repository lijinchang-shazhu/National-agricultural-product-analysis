import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def process_data(series, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(series) - n_future + 1):
        X.append(series[i - n_past:i])
        y.append(series[i + n_future - 1])
    return np.array(X), np.array(y)

def train_and_predict(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)
    results = []
    scaler = MinMaxScaler()

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            series = df['平均价'].values

            if len(series) < 30:
                print(f"Skipping {filename} due to insufficient data points.")
                continue

            series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            X, y = process_data(series, n_past=14, n_future=7)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
            
            model = create_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

            # Generate predictions
            future_x = process_data(np.append(series, [np.nan]*7), 14, 7)[0][-1:]
            predictions = model.predict(future_x).flatten()
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            y_pred = model.predict(X_test).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Save predictions
            full_series = np.concatenate([scaler.inverse_transform(series.reshape(-1, 1)).flatten(), predictions])
            results_df = pd.DataFrame({'ID': range(len(full_series)), '平均价': full_series})
            results_df.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)

            results.append([filename, len(series), mae, mse, r2])

    # Save all results
    summary_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    summary_df.to_csv(results_file, index=False)

# 设置输入输出路径和结果文件路径
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\\LSTM'
results_file = 'I:\全国农产品分析\\norm_LSTM.csv'

train_and_predict(input_directory, output_directory, results_file)
