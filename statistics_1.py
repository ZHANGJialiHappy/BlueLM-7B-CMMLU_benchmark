import os
import csv

# 定义统计函数
def process_csv(folder_name, output_folder):
    # 获取该文件夹下所有csv文件
    csv_files = [f for f in os.listdir(folder_name) if f.endswith('.csv')]
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 统计文件名
    output_file = os.path.join(output_folder, f'statistic_{os.path.basename(output_folder)}.csv')
    
    # 打开统计文件，准备写入统计结果
    with open(output_file, 'w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        # 写入CSV表头
        writer.writerow(['主题', '总题数', '正确回答数', '正确率 (%)'])
        
        # 遍历每个csv文件
        for csv_file in csv_files:
            total_questions = 0
            correct_answers = 0
            topic_name = csv_file.replace('results_', '').replace('.csv', '')
            
            # 读取csv文件内容
            with open(os.path.join(folder_name, csv_file), 'r') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    total_questions += 1
                    correct_answer = row[-2].strip()  # 倒数第二列是正确答案
                    model_answer = row[-1].strip()  # 倒数第一列是模型的答案
                    
                    if correct_answer == model_answer:
                        correct_answers += 1
            
            # 计算正确率
            accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            
            # 将统计结果写入输出csv文件
            writer.writerow([topic_name, total_questions, correct_answers, round(accuracy, 2)])

# 文件夹路径定义
input_folders = ['BlueLM-7B-Base-5-shot-culture', 'BlueLM-7B-Base-5-shot-non-culture']
output_folders = ['statistic_BlueLM-7B-Base-5-shot-culture', 'statistic_BlueLM-7B-Base-5-shot-non-culture']

# 遍历输入文件夹并处理
for input_folder, output_folder in zip(input_folders, output_folders):
    process_csv(input_folder, output_folder)

print("统计结果已生成！")
