# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, redirect, url_for,jsonify
from pro import getdata,getpredict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import re

app = Flask(__name__)
# 模拟用户数据，实际情况下应从数据库中获取
users = {
    'user1': {'password': 'aaa'},
    'user2': {'password': 'bbb'}
}
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if str(username) in users and users[str(username)]['password'] == password:
            # return redirect(url_for('query'))
            return render_template("start.html")
        else:
            return render_template('login.html', error="Invalid username or password. Please try again.")
    return render_template('login.html')

@app.route("/AlgorithmNorm",methods=['GET','POST'])
def AN():
    return render_template("AlgorithmNorm.html")

@app.route("/AlgorithmPredict",methods=['GET','POST'])
def AP():
    return render_template("AlgorithmPredict.html")

@app.route('/index', methods=['GET', 'POST'])
def query():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('index.html', dict_return=dict_return)
    else:
        dict_return = getdata('北方江米')                       #默认初始页面
        return render_template('index.html', dict_return=dict_return)

@app.route('/chart1', methods=['GET', 'POST'])
def chart1():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('chart1.html', dict_return=dict_return)
    else:
        dict_return = getdata('北方江米')                       #默认初始页面
        return render_template('chart1.html', dict_return=dict_return)

@app.route('/chart2', methods=['GET', 'POST'])
def chart2():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('chart2.html', dict_return=dict_return, product=product)
    else:
        dict_return = getdata('北方江米')  # 默认初始页面
        return render_template('chart2.html', dict_return=dict_return, product='北方江米') # 将默认产品名称传递给模板

@app.route('/chart3', methods=['GET', 'POST'])
def chart3():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('chart3.html', dict_return=dict_return, product=product)
    else:
        dict_return = getdata('北方江米')  # 默认初始页面
        return render_template('chart3.html', dict_return=dict_return, product='北方江米') # 将默认产品名称传递给模板

@app.route('/chart4', methods=['GET', 'POST'])
def chart4():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('chart4.html', dict_return=dict_return, product=product)
    else:
        dict_return = getdata('北方江米')  # 默认初始页面
        return render_template('chart4.html', dict_return=dict_return, product='北方江米') # 将默认产品名称传递给模板

@app.route('/chart5', methods=['GET', 'POST'])
def chart5():
    if request.method == "POST":
        product = request.form.get("product")
        dict_return = getdata(product)
        return render_template('chart5.html', dict_return=dict_return, product=product)
    else:
        dict_return = getdata('北方江米')  # 默认初始页面
        return render_template('chart5.html', dict_return=dict_return, product='北方江米') # 将默认产品名称传递给模板


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        product = request.form.get("product")
        predictdata = getpredict(product)
        return render_template('predict.html', predictdata=predictdata, product=product)
    else:
        predictdata = getpredict('金龙鱼大豆油')
        return render_template('predict.html', predictdata=predictdata, product='金龙鱼大豆油')


@app.route('/tab_clicked', methods=['POST'])
def handle_tab_click():
    data = request.get_json()  # 获取 JSON 数据
    tab_name = data['tabName']  # 从 JSON 中提取 tabName
    directory_path = f"I:\\全国农产品分析\\{tab_name}"  # 拼接路径

    # 确保路径存在
    if not os.path.exists(directory_path):
        return jsonify({'status': 'error', 'message': 'Directory does not exist'}), 404

    # 获取目录下所有文件名
    filenames = os.listdir(directory_path)
    # 处理文件名并去除不需要的部分
    processed_names = [re.sub(r'predict_|_|价格|.csv', '', filename) for filename in filenames]
    app.logger.info(f"数据为：{processed_names}")
    # 返回处理后的文件名列表
    return jsonify({'status': 'success', 'models': processed_names})

@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.get_json()
    tab_name = data['tabName']
    model_name = data['modelName']
    file_path = f"I:\\全国农产品分析\\{tab_name}\\predict_{model_name}_价格.csv"
    app.logger.info(f"路径为{file_path}")
    try:
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)  # 将 NaN 替换为0

        df['平均价'] = df['平均价'].apply(lambda x: round(abs(x), 2))  # 取绝对值并保留两位小数
        last_7 = df.tail(7)['平均价'].tolist()
        data = {
            "ids": df['ID'].tolist(),
            "prices": df['平均价'].tolist(),
            "forecast": last_7
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    data = request.get_json()
    algorithm_name = data['algorithmName']  # This should match the tab text content sent from frontend
    file_path = f"I:/全国农产品分析/Norm/{algorithm_name}.csv"
    print(file_path)
    try:
        df = pd.read_csv(file_path)

        # Process and filter the data as needed
        # We will process four parts of the data, each part for a specific visualization

        # Part 1: Average Absolute Error (If -100, skip)
        mse_df = df[['数据集名称', '平均绝对误差']].replace(-100, pd.NA).dropna()

        # Part 2: Root Mean Square Error (If -100, skip)
        rmse_df = df[['数据集名称', '均方根误差']].replace(-100, pd.NA).dropna()

        # Part 3: Coefficient of Determination (If -100, skip)
        r2_df = df[['数据集名称', '决定系数']].replace(-100, pd.NA).dropna()

        # Part 4: Forecasting Accuracy
        # Evaluate performance categories based on the given criteria
        def evaluate_performance(row):
            if row['决定系数'] > 0 and row['均方根误差'] < 10 and row['平均绝对误差'] < 10:
                return '优秀'
            elif row['决定系数'] == 0 and row['均方根误差'] < 10 and row['平均绝对误差'] < 10:
                return '良好'
            elif row['决定系数'] < 0:
                return '差'
            return '其他'

        df['预测准确率'] = df.apply(evaluate_performance, axis=1)
        pr_counts = df['预测准确率'].value_counts().to_dict()

        # Return the processed data as JSON
        return jsonify({
            'mse': mse_df.to_dict(orient='records'),
            'rmse': rmse_df.to_dict(orient='records'),
            'r2': r2_df.to_dict(orient='records'),
            'pr': pr_counts
        })

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")  # 输出详细的错误信息到日志

        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



