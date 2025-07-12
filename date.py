import pandas as pd

# 加载数据
df = pd.read_csv('I:\全国农产品分析\农产品价格情况\金龙鱼大豆油价格.csv')

# 显示日期列的一些样本数据
# print(df['发布日期'].head())
# 转换日期格式，错误的日期转为 NaT
df['发布日期'] = pd.to_datetime(df['发布日期'], errors='coerce', format='%Y-%m-%d')

# 查看转换后的数据
print(df['发布日期'])
