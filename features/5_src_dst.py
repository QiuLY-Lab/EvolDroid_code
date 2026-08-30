import csv

# 输入文件路径
input_file = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/malradar_tgn_dataset_edges.csv"

# 初始化集合存储节点
first_col_nodes = set()  # 第一列节点集合
second_col_nodes = set()  # 第二列节点集合

# 读取CSV文件并收集节点
with open(input_file, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头行

    for row in reader:
        if len(row) >= 2:  # 确保行有至少两列
            first_node = row[0]
            second_node = row[1]
            first_col_nodes.add(first_node)
            second_col_nodes.add(second_node)

# 计算统计结果
first_col_count = len(first_col_nodes)  # 第一列不同节点数
second_col_count = len(second_col_nodes)  # 第二列不同节点数
only_first = len(first_col_nodes - second_col_nodes)  # 只在第一列出现在节点数
only_second = len(second_col_nodes - first_col_nodes)  # 只在第二列出现在节点数

# 输出结果
print(f"第一列不同节点数量: {first_col_count}")
print(f"第二列不同节点数量: {second_col_count}")
print(f"只出现在第一列的节点数量: {only_first}")
print(f"只出现在第二列的节点数量: {only_second}")