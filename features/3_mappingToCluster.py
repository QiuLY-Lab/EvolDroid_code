import pickle
from pathlib import Path

def find_all_files(directory):
    """
    递归查找目录下所有文件，返回Path列表
    """
    files = []
    for item in directory.iterdir():
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(find_all_files(item))
    return files

def mapping():
    # 加载 method -> cluster 映射
    with open('./res/method_cluster_mapping_2000.pkl', 'rb') as f:
        method_cluster = pickle.load(f)

    # 遍历所有 apk 对应的 API 调用文件
    # source_dir = Path(r"/home/qly/mycode/CADE_API_processed")
    source_dir = Path(r"/home/qly/mycode/malradar/processed_API_new")
    app_files = find_all_files(source_dir)

    data_dict = {}  # 保存 apk_name -> vec
    for app_file in app_files:
        apk_name = app_file.stem.split("_")[0]  # 文件名（去掉扩展名）
        with open(app_file, "r", encoding="utf-8") as f:
            apis = [line.strip() for line in f]

        vec = [0] * 2000
        for a in apis:
            if a in method_cluster:
                vec[method_cluster[a]] = 1
        data_dict[apk_name] = vec

    # 保存整个数据字典到 pickle 文件
    # with open("./CADE_API_Graph_new_2.pkl", "wb") as f:
    with open("./malradar_API_Graph.pkl", "wb") as f:
        pickle.dump(data_dict, f)

    # print("✅ 保存完成：./CADE_API_Graph_new_2.pkl")
    print("✅ 保存完成：./malradar_API_Graph.pkl")

if __name__ == '__main__':
    mapping()

# import pickle
#
# # 加载整个 API 图向量字典
# with open("./output/API_Graph.pkl", "rb") as f:
#     api_graph_dict = pickle.load(f)
#
# # 使用示例
# for apk_name, vec in api_graph_dict.items():
#     print(apk_name, vec[:20])  # 只打印前 20 个元素查看