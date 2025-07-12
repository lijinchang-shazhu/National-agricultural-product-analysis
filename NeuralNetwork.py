import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def process_files(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)
    results = []

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            series = df['平均价']

            if len(series) < 20:
                print(f"Skipping {filename} due to insufficient data points.")
                continue

            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=7, random_state=42, shuffle=False)

            # Create and train model
            model = create_model(1)
            model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

            # Predict future values
            future_X = scaler.transform(np.arange(len(series), len(series) + 7).reshape(-1, 1))
            predictions = model.predict(future_X).flatten()
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Evaluate model
            y_pred_scaled = model.predict(X_test).flatten()
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save predictions and original data
            full_data = pd.DataFrame({
                'ID': np.arange(len(series) + 7),
                '平均价': np.concatenate([series, predictions])
            })
            full_data.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)

            # Append results
            results.append([filename, len(series), mae, mse, r2])

    # Save all results
    results_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    results_df.to_csv(results_file, index=False)

# Set paths
# 设置输入输出路径和结果文件路径
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\\NeuralNetwork'
results_file = 'I:\全国农产品分析\\norm_NeuralNetwork.csv'

process_files(input_directory, output_directory, results_file)
