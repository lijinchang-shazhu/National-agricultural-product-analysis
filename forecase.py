import pandas as pd
from datetime import timedelta
import numpy as np
from keras import Input
from sklearn.preprocessing import MinMaxScaler # sklearn安装的包名,scikit-learn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# 1. 加载数据
veg = '乐初非转基因大豆油'
data = pd.read_csv('./农产品价格情况/{}价格.csv'.format(veg), encoding='utf-8')
date_format = '%Y-%m-%d'
data['发布日期'] = pd.to_datetime(data['发布日期'], format=date_format)
date_back = data['发布日期']
# 2. 创建未来7天的日期
last_date = data['发布日期'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]


# 3. 创建包含未来日期的数据框
future_data = pd.DataFrame({'发布日期': future_dates})

# 4. 预处理数据
# 将日期转换为数值
data['发布日期'] = (data['发布日期'] - pd.to_datetime(data['发布日期'].min(), format=date_format)).dt.days

# 将数据缩放到0和1之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['发布日期', '平均价']])
# 创建训练数据集
X_train = []
y_train = []
for i in range(7, len(scaled_data)):
    X_train.append(scaled_data[i-7:i, :])
    y_train.append(scaled_data[i, :1])
X_train, y_train = np.array(X_train), np.array(y_train)


# 5. 构建并训练深度学习模型
model = Sequential()
# 定义输入的形状
input_shape = (X_train.shape[1], 2)
# 使用Sequential模型，并在第一层使用Input对象
model = Sequential()
model.add(Input(shape=input_shape))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=7))


model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 损失可视化
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# 6. 预测未来7天的价格
# 基于最后7天的数据进行预测
last_7_days = scaled_data[-7:]
X_future = np.array([last_7_days])

# Reshape数据以符合LSTM模型的输入要求
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 2))

# 使用模型进行预测
predicted_prices_scaled = model.predict(X_future)
# 将预测价格的形状reshape以匹配last_7_days[:, 0]的形状
predicted_prices_scaled_reshaped = predicted_prices_scaled.reshape(-1, 1)

# 将last_7_days[:, 0]与预测的价格合并
predicted_prices_combined = np.concatenate((last_7_days[:, 0].reshape(-1, 1), predicted_prices_scaled_reshaped), axis=1)
# 对合并后的价格进行反向缩放以获得原始范围
predicted_prices = scaler.inverse_transform(predicted_prices_combined)[:, 1]

# 提取未来7天的预测价格
predicted_prices_future = predicted_prices[-7:]

# 创建一个DataFrame来存储未来日期和预测价格
future_price_df = pd.DataFrame({'日期': future_dates, '价格': predicted_prices_future})

# 打印未来7天的预测价格
print("未来7天的预测价格:")
print(future_price_df)

# 将日期转换回原始格式
print("data['发布日期'].min(), format=date_format)\n   ",data['发布日期'].min())
print("pd.to_timedelta(data['发布日期'], unit='D')\n",pd.to_timedelta(data['发布日期']))
#data['发布日期'] = pd.to_datetime(data['发布日期'].min(), format=date_format) + pd.to_timedelta(data['发布日期'])

# 显示修改后的 DataFrame
# 历史数据和预测结果曲线图
plt.plot(date_back[-30:], data['平均价'][-30:], label='Historical Data')
plt.plot(future_price_df['日期'], future_price_df['价格'], 'ro',label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Historical Data and Predicted Prices')
plt.legend()
plt.show()
