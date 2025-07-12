import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 示例数据
index = pd.date_range(start='2020-01-01', periods=36, freq='M')
data = np.random.normal(loc=10, scale=2, size=36)  # 随机生成的示例数据
series = pd.Series(data, index=index)

# 设置并拟合模型
model = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add')
fit = model.fit()

# 预测未来12个月
forecast = fit.forecast(steps=12)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(series, label='Original Data')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Holt-Winters Forecast')
plt.legend()
plt.show()
