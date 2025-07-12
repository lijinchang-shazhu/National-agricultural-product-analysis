import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_predict(input_directory, output_directory, results_file):
    os.makedirs(output_directory, exist_ok=True)  # 确保输出目录存在
    results = []

    # 遍历指定目录下的所有CSV文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)

            # 准备数据
            df['ID'] = range(len(df))  # 添加一个简单的索引作为特征
            X = df[['ID']]  # 特征矩阵
            y = df['平均价']  # 目标变量

            # 检查数据量是否足够
            if len(df) <= 7:
                print(f"数据量不足，无法进行训练和预测: {filename}")
                continue

            # 拆分数据集用于训练和预测
            X_train, X_future = X[:-7], X[-7:]
            y_train = y[:-7]

            # 创建并训练Lasso模型
            model = Lasso(alpha=0.1)
            model.fit(X_train, y_train)

            # 进行预测
            y_pred = model.predict(X)
            y_future = model.predict(X_future)

            # 评估模型
            mae = mean_absolute_error(y_train, y_pred[:-7])
            rmse = np.sqrt(mean_squared_error(y_train, y_pred[:-7]))
            r2 = r2_score(y_train, y_pred[:-7])

            # 保存拟合与预测数据
            future_df = pd.DataFrame({'ID': range(len(df), len(df) + 7), '平均价': y_future})
            combined_df = pd.concat([df, future_df])
            combined_df.to_csv(os.path.join(output_directory, f"predict_{filename}"), index=False)
            
            # 记录结果
            results.append([filename, len(y_train), mae, rmse, r2])

    # 保存结果到CSV文件
    results_df = pd.DataFrame(results, columns=['数据集名称', '所使用数据量条数', '平均绝对误差', '均方根误差', '决定系数'])
    results_df.to_csv(results_file, index=False)

# 设置输入输出路径和结果文件路径
input_directory = 'I:\全国农产品分析\数据整理'
output_directory = 'I:\全国农产品分析\LassoRegression'
results_file = 'I:\全国农产品分析\\norm_Lasso.csv'

# 执行函数
train_and_predict(input_directory, output_directory, results_file)
