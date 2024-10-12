
import os
import csv

# 文件夹名称
input_folders = ['a_answers', 'b_answers']
output_folders = ['a_accuracy', 'b_accuracy']

# 创建输出文件夹
for output_folder in output_folders:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# 遍历输入文件夹和CSV文件
for input_folder, output_folder in zip(input_folders, output_folders):
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # 读取CSV文件
            file_path = os.path.join(input_folder, filename)
            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                correct_count = 0
                total_count = 0
                
                # 遍历CSV中的每一行
                for row in csv_reader:
                    if len(row) < 3:
                        continue  # 跳过格式不正确的行
                    total_count += 1
                    correct_answer = row[-2]  # 倒数第二列是正确答案
                    model_answer = row[-1]  # 倒数第一列是模型的答案
                    if correct_answer == model_answer:
                        correct_count += 1
                
                # 计算准确率
                accuracy_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0

                # 准备输出数据
                output_data = [
                    ['Correct Count', 'Total Count', 'Accuracy Percentage'],
                    [correct_count, total_count, accuracy_percentage]
                ]

                # 写入结果到新的CSV文件
                output_file_path = os.path.join(output_folder, filename)
                with open(output_file_path, mode='w', newline='', encoding='utf-8') as output_csvfile:
                    csv_writer = csv.writer(output_csvfile)
                    csv_writer.writerows(output_data)

print("Accuracy calculations completed.")
