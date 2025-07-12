import os
import pandas as pd

def process_and_save_data(input_directory, output_directory):
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 读取指定目录下的所有CSV文件
    all_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith('.csv')]
    all_data = []

    # 合并所有CSV文件到一个DataFrame
    for file in all_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)

    # 按品名分组并保存为CSV文件
    for product_name, group in combined_data.groupby('品名'):
        # 定义输出文件路径
        output_file_path = os.path.join(output_directory, f"{product_name}_价格.csv")
        # 保存CSV
        group.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# 指定输入和输出目录
input_directory = 'I:\全国农产品分析\蔬菜价格'  # 这里替换为你的输入目录路径
output_directory = 'I:\全国农产品分析\数据整理'  # 这里替换为你的输出目录路径

# 调用函数处理并保存数据
process_and_save_data(input_directory, output_directory)
