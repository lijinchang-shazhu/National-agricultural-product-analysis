# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import json
import os
import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras import Input

import matplotlib.pyplot as plt


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

def getdata(product):
    dict_return = {}
    #处理好的文件路径
    path = "I:\全国农产品分析\农产品价格情况/"
    file = path+str(product)+"价格.csv"
    
    # 同城两个商品加入雷达图作对比
    file1 = path +  "/秋然香米价格.csv"
    file2 = path +  "/南方江米价格.csv"


    data = pd.read_csv(file,encoding='utf8`')
    data_1 = pd.read_csv(file1,encoding='utf8')
    data_2 = pd.read_csv(file2,encoding='utf8')

    data_1 = list(data_1[u'平均价'][:8])
    data_2 = list(data_2[u'平均价'][:8])
    data_3 = list(data[u'平均价'][:8])

    month = list(data[u'发布日期'])
    high_price = list(data[u'最高价'])
    low_price = list(data[u'最低价'])
    price = list(data[u'平均价'])

    # 以下为将处理好的数据加入字典
    dict_return['date'] = month
    print(dict_return)
    dict_return['high_price'] = high_price
    dict_return['price'] = price


    dict_return['low_price'] = list(low_price)

    jsonList = []
    for i in range(0, len(low_price)):
        jsonList.append({ 'name':month[i],'value': low_price[i]})
    data0 = json.dumps(jsonList,cls=MyEncoder,ensure_ascii = False)
    data0 = json.loads(data0)
    dict_return['data0'] = data0

    radar0= []
    radar1 = [{'value':data_1,'name':month[0]}]
    radar2 = [{'value':data_2,'name':month[5]}]
    radar3 = [{'value':data_3,'name':month[6]}]

    for j in range(0, 8):
            radar0.append({ 'name':month[j],'max': max(high_price)+1})
    data1 = json.dumps(radar0,cls=MyEncoder,ensure_ascii = False)
    data1 = json.loads(data1)
    dict_return['radar0'] = data1

    dict_return['radar1'] = radar1
    dict_return['radar2'] = radar2
    dict_return['radar3'] = radar3

    # 滚动图
    item = []
    for i in range(0, len(low_price)):
        item.append({'date':month[i],'price': price[i],'high_price':high_price[i],'low_price':low_price[i]})
    data_tb = json.dumps(item,cls=MyEncoder,ensure_ascii = False)
    data_tb = json.loads(data_tb)
    dict_return['data_tb'] = data_tb
    dict_return['n'] = product

    return dict_return
def getpredict(product):
    predictdata = {}
    # 1. 加载数据
    data = pd.read_csv('I:/全国农产品分析/农产品价格情况/{}价格.csv'.format(product), encoding='utf-8')
    date_format = '%Y-%m-%d'
    data['发布日期'] = pd.to_datetime(data['发布日期'], errors='coerce',format=date_format)
    date_back = data['发布日期']
    # 2. 创建未来7天的日期
    last_date = data['发布日期'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]


    # 3. 创建包含未来日期的数据框
    future_data = pd.DataFrame({'发布日期': future_dates})

    # 4. 预处理数据
    # 将日期转换为数值
    data['发布日期'] = (data['发布日期'] - pd.to_datetime(data['发布日期'].min(), errors='coerce', format='%Y-%m-%d')).dt.days

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
        # 将训练损失存入predictdata
    predictdata['loss'] = [round(i,3) for i in history.history['loss']]
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
    predicted_prices_combined = np.concatenate((last_7_days[:, 0].reshape(-1, 1), predicted_prices_scaled_reshaped),
                                               axis=1)

    # 对合并后的价格进行反向缩放以获得原始范围
    predicted_prices = scaler.inverse_transform(predicted_prices_combined)[:, 1]

    # 提取未来7天的预测价格
    predicted_prices_future = predicted_prices[-7:]

    # 创建一个DataFrame来存储未来日期和预测价格
    future_price_df = pd.DataFrame({'日期': future_dates, '价格': predicted_prices_future})

    # 打印未来7天的预测价格
    print("未来7天的预测价格:")
    print(future_price_df)
    # print(predicted_prices[0:-7])
    price_back=data['平均价']
    # 将日期转换回原始格式
    data['发布日期'] = pd.to_datetime(data['发布日期'].min(),errors='coerce', format='%Y-%m-%d') + pd.to_timedelta(data['发布日期'], unit='D')

    future_price_df['日期'] = pd.to_datetime(future_price_df['日期'])
    lida = [i.strftime('%Y-%m-%d') for i in future_dates]
    predictdata['rawdate']=date_back.tolist()
    predictdata['newdate']=date_back.tolist()+lida
    predictdata['rawprice']=price_back.tolist()
    predictdata['newprice']=price_back.tolist()+[round(num,1) for num in predicted_prices_future]
    predictdata['epochs']=[i for i in range(1,len(history.history['loss'])+1)]
    predictdata['fudate']=lida
    predictdata['fuprice']=[round(i,3) for i in predicted_prices[-7:]]
    # 假设您有一个包含价格数据的列表 raw_prices
    raw_prices = predictdata['rawprice']

    # 统计每个价格点的频数
    price_frequency = {}
    for price in raw_prices:
        if price not in price_frequency:
            price_frequency[price] = 0
        price_frequency[price] += 1

    # 对价格频数字典按价格进行排序
    sorted_price_frequency = sorted(price_frequency.items())

    # 构建指定格式的数组
    price_list = []
    fre_list = []
    for price, frequency in sorted_price_frequency:
        price_list.append(price)
        fre_list.append(frequency)
    predictdata['price_list']=price_list
    predictdata['fre_list']=fre_list
    # 滚动图
    item0 = []
    for i in range(0, len(lida)):
        item0.append({'date':lida[i],'price': predictdata['fuprice'][i]})
    data_tb = json.dumps(item0,cls=MyEncoder,ensure_ascii = False)
    data_tb = json.loads(data_tb)
    predictdata['fup'] = data_tb
    predictdata['n'] = product

    return predictdata


