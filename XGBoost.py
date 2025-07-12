import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def process_files(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)
    results = []

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            series = df['平均价']

            if len(series) < 30:
                print(f"Skipping {filename} due to insufficient data points.")
                continue

            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=7, random_state=42, shuffle=False)

            model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror',
                                 n_jobs=-1, reg_lambda=1, reg_alpha=0.5)
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=10, verbose=False)

            predictions = model.predict(np.arange(len(series), len(series) + 7).reshape(-1, 1))

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            full_data = pd.DataFrame({'ID': np.arange(len(series) + 7), '平均价': np.concatenate([y, predictions])})
            full_data.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)

            results.append([filename, len(series), mae, mse, r2])

    results_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    results_df.to_csv(results_file, index=False)

# 设置输入输出路径和结果文件路径
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\XGBoostRegression'
results_file = 'I:\全国农产品分析\\norm_XGBoost.csv'

process_files(input_directory, output_directory, results_file)
