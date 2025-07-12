import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def adf_test(series):
    """Perform Dickey-Fuller test and return the p-value."""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[1]  # p-value
    except ValueError:
        # 如果数据是常数，返回 None
        return None

def select_best_lag(train, max_lag=30):
    """Determine the best lag for the AutoReg model based on AIC."""
    best_lag = 1
    best_aic = float('inf')
    for lag in range(1, min(max_lag, len(train) // 2 - 1) + 1):
        try:
            model = AutoReg(train, lags=lag)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_lag = lag
        except ValueError:
            # If the model cannot be estimated, return None
            return None
    return best_lag

def train_and_predict(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)
    results = []

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            series = df['平均价']

            if series.nunique() == 1 or len(series) <= 2:  # 检查数据是否足够
                print(f"Skipping {filename} as the data is constant or too small.")
                results.append([filename, len(series), np.nan, np.nan, np.nan])
                continue

            p_value = adf_test(series)
            if p_value is None or p_value > 0.05:
                series = series.diff().dropna()  # 差分

            if series.empty or series.nunique() == 1:
                print(f"No valid data available after differencing for {filename}.")
                # results.append([filename, len(series), np.nan, np.nan, np.nan])
                continue

            best_lag = select_best_lag(series)
            if best_lag is None:
                print(f"Cannot estimate model for {filename} due to insufficient data after differencing.")
                continue

            train = series[:-7]
            test = series[-7:]

            try:
                model = AutoReg(train, lags=best_lag)
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(train), end=len(train) + 6, dynamic=True)

                mae = mean_absolute_error(test, predictions)
                mse = mean_squared_error(test, predictions)
                r2 = r2_score(test, predictions)

                combined_data = pd.concat([series, pd.Series(predictions, index=test.index)])
                combined_data = combined_data.to_frame(name='平均价')
                combined_data['ID'] = np.arange(len(combined_data))
                combined_data.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)

                results.append([filename, len(series), mae, mse, r2])
            except Exception as e:
                print(f"Failed to process {filename} due to: {e}")
                continue

    results_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    results_df.to_csv(results_file, index=False)

# Set paths
# 设置输入输出路径和结果文件路径
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\\AutoRegression'
results_file = 'I:\全国农产品分析\\norm_AutoRegressioin.csv'

train_and_predict(input_directory, output_directory, results_file)
