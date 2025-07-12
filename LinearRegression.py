import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_predict(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)  # 确保输出目录存在
    results = []

    # 遍历指定目录下的所有CSV文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            # print(type(file_path))
            df = pd.read_csv(file_path,encoding="utf-8-sig")

            # 确保数据是连续的数字序列
            df['ID'] = range(len(df))
            
            # 仅使用包含平均价的行
            X = df['ID'].values.reshape(-1, 1)
            y = df['平均价'].values
            
            # 使用线性回归模型
            model = LinearRegression()
            model.fit(X, y)

            # 预测未来七个值
            future_preds = model.predict(np.array(range(len(df), len(df) + 7)).reshape(-1, 1))
            
            # 模型评估
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            # 保存拟合与预测数据
            future_df = pd.DataFrame({
                'ID': range(len(df), len(df) + 7),
                '平均价': future_preds
            })
            combined_df = pd.concat([df, future_df])
            combined_df.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)
            
            # 添加结果到列表
            results.append([filename, len(df), mae, rmse, r2])

    # 保存结果到CSV文件
    results_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    results_df.to_csv(results_file, index=False)

# 设置输入输出路径和结果文件
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\LinearRegression'
results_file = 'I:\全国农产品分析\\norm_Linear.csv'

# 调用函数
train_and_predict(input_directory, output_directory, results_file)
