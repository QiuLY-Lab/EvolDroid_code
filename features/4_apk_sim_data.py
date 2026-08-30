import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

def to_unix(ts):
    try:
        return int(datetime.strptime(ts, "%Y-%m-%d").timestamp())
        # .timestamp() 把 datetime 转成 Unix 时间戳（从 1970-01-01 00:00:00 UTC 开始的秒数）。
    except:
        return 0

# =========================================================
# 1️⃣ 加载训练 & 测试 APK 向量
# =========================================================
# train_path = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/advanced_API_Graph_newdataset.pkl"
train_path = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/malradar_API_Graph.pkl"
test_path = "/home/cpl/qlycode/ExtractAPIGraphFeature/ExtractAPIGraph/CADE_test_API_Graph.pkl"  # 如果没有测试集可以去掉

with open(train_path, "rb") as f:
    train_vectors = pickle.load(f)

# ✅ train_vectors 是 dict: {apk_name: [0,1,0,...]}
# 如果有测试集也加载并合并
if os.path.exists(test_path):
    with open(test_path, "rb") as f:
        test_vectors = pickle.load(f)
    apk_vectors = {**train_vectors, **test_vectors}
else:
    print("没有测试集！")
    apk_vectors = train_vectors
print(f"✅ 共加载 {len(apk_vectors)} 个 APK 向量")

# =========================================================
# 2️⃣ 加载时间戳文件（txt 格式，每行：apk_name timestamp）
# =========================================================
# timestamp_file = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/CADE_apk_family_ts_new_3.txt"
timestamp_file = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/malradar_family_ts_new.txt"
apk_timestamps = {}
apk_families = {}  # 🆕 新增：存储每个apk的family
family_to_id = {}  # 🆕 新增：family → 数字ID
next_id = 0

with open(timestamp_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        # parts ['trojan_kyview', '20EC5BCEF6834ACDE1F881ADBFB3E35DADFE2D60CA18E064C5D7DA1DC1882250.apk', '2016-10-19', '15:58:32']
        if len(parts) == 4:
            family, apk, ts_date,ts_time = parts
            apk_name = apk.split(".")[0]
            apk_timestamps[apk_name] = ts_date

            # 🆕 给family分配一个数字编号
            if family not in family_to_id:
                family_to_id[family] = next_id
                next_id += 1
            apk_families[apk_name] = family_to_id[family]
print(f"✅ 加载 {len(apk_timestamps)} 个 APK 时间戳")
print(f"✅ 检测到 {len(family_to_id)} 个家族类别")

# =========================================================
# 3️⃣ 生成 APK 列表和对应矩阵
# =========================================================
apk_names = list(apk_vectors.keys())
# apk_names = [os.path.basename(a) for a in apk_vectors.keys()]
vectors = np.stack([apk_vectors[a] for a in apk_names])
print("apk_names[0]",apk_names[0])
for a in apk_names:
    print("a",a)
    break

num_apk = len(apk_names)
vector_dim = vectors.shape[1]
print("family_to_id",family_to_id)

# 计算余弦相似度矩阵
# sim_matrix = cosine_similarity(vectors)
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
# 计算欧几里得距离矩阵
dist_matrix = euclidean_distances(vectors)
# 将“距离”转为“相似度” （距离越小 → 相似度越大）
sim_matrix = 1 / (1 + dist_matrix)

# # 输出相似度分布
# # sim_matrix 是你的相似度矩阵
# sim_values = sim_matrix.flatten()
# # 只取 上三角, k=1 是去掉对角线
# sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
# bins = np.arange(0, 1.1, 0.05)
# hist, edges = np.histogram(sim_values, bins=bins)
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,6))

# # 单组小提琴图
# sns.violinplot(data=[sim_values], palette=['skyblue'],inner=None)

# plt.xticks([0], ['Similarity'])  # 只有一组时
# plt.ylabel("Similarity")
# plt.title("Similarity Distribution - Violin Plot")
# plt.grid(axis='y', linestyle='--')
# plt.show()

# # 归一化 -> 比例
# proportion = hist / np.sum(hist)
#
# # x轴标签（区间）
# labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))]
#
# # 画柱状图
# plt.figure()
# plt.bar(labels, proportion)
#
# plt.xlabel('Similarity Interval')
# plt.ylabel('Proportion')
# plt.title('Similarity Distribution per Interval')
#
# plt.xticks(rotation=45)
# plt.grid(axis='y')
#
# # 如果想显示具体比例
# for i, v in enumerate(proportion):
#     plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
# plt.savefig("CDF.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.figure(figsize=(10,6))
# sns.kdeplot(sim_values, bw=0.5)  # bw_adjust 控制平滑程度
# plt.xlabel("Similarity")
# plt.ylabel("Density")
# plt.title("Similarity Density Plot (KDE)")
# plt.grid(True, axis='y', linestyle='--')
# 保存
# np.save("malradar-sim_values.npy", sim_values)
# plt.savefig("malradar-violin.png", dpi=300, bbox_inches="tight")
# plt.show()
# # 输出相似度分布
# for i in range(len(hist)):
#     print(f"{edges[i]:.2f}-{edges[i+1]:.2f}: {hist[i]}")


timestamps = np.array([to_unix(ts) for ts in apk_timestamps.values()])
min_ts = timestamps.min()
# 所有时间转成相对秒数（从0开始）
apk_timestamps_rel = {k: v - min_ts for k, v in zip(apk_timestamps.keys(), timestamps)}

# =========================================================
# 4️⃣ 构建边（相似度 > 阈值）
# =========================================================
edges = []
threshold = 0.8
for i in range(num_apk):  # 节点编号从0开始
    for j in range(i + 1, num_apk):
        sim = sim_matrix[i, j]
        if sim >= threshold:
            apk_i, apk_j = apk_names[i], apk_names[j] # 没有.apk
            ts_i = apk_timestamps_rel.get(apk_i, "1970-01-01")  # 如果字典中没有这个键（例如 APK 没有记录时间），就返回默认时间 "1970-01-01"。
            ts_j = apk_timestamps_rel.get(apk_j, "1970-01-01")
            # min_ts 1009814400
            # apk_timestamps_rel.get(apk_j, '1970-01-01') 450748800
            # apk_timestamps.get(apk_j, '1970-01-01') ts_j 1460563200
            # edge_time = max(to_unix(ts_i), to_unix(ts_j))
            edge_time = max(ts_i, ts_j)

            # 确保时间更早的在前面
            if ts_i <= ts_j:
                src, dst = i, j
                edge_time = ts_j  # 边的时间用较晚的那个节点时间
                family_id = apk_families.get(apk_j, -1)
            else:
                src, dst = j, i
                edge_time = ts_i
                family_id = apk_families.get(apk_i, -1)
            edges.append((src, dst, edge_time, family_id, sim))
print(f"✅ 构建 {len(edges)} 条边（相似度 > {threshold}）")

# =========================================================
# 🧩 先按时间排序并切分 train / val / test
# =========================================================
print("🔧 正在进行时间排序并划分 train/val/test...")
apk_time_pairs = []
for idx, apk in enumerate(apk_names):
    ts = apk_timestamps_rel.get(apk, 0)
    apk_time_pairs.append((idx, apk, ts))
# 按时间升序排序
apk_time_pairs.sort(key=lambda x: x[2])
num_train = int(0.85 * num_apk)
num_val = int(0.85 * num_apk)

train_idx = set([p[0] for p in apk_time_pairs[:num_train]])
val_idx = set([p[0] for p in apk_time_pairs[num_train:num_val]])
test_idx = set([p[0] for p in apk_time_pairs[num_val:]])
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# =========================================================
# 🧩 找孤立节点
# =========================================================
all_nodes = set(range(num_apk))
connected = set()
for u, v, _, _, _ in edges:
    connected.add(u)
    connected.add(v)
isolated_nodes = list(all_nodes - connected)
print(f"🔍 共检测到 {len(isolated_nodes)} 个孤立节点")

# =========================================================
# 计算 family 平均 embedding（严格使用 train + val）
# =========================================================
print("🔧 正在计算 family 平均 embedding（仅 train + val）...")
family_embeddings = {}
for fam_id in family_to_id.values():
    # 只用 train + val
    members = [
        apk_names[idx]
        for idx in (train_idx | val_idx)
        if apk_families.get(apk_names[idx], -1) == fam_id
    ]
    if len(members) > 0:
        vecs = np.stack([apk_vectors[apk] for apk in members])
        family_embeddings[fam_id] = vecs.mean(axis=0)
print(f"已为 {len(family_embeddings)} 个 family 计算平均 embedding")

# =========================================================
# 🧩 对训练集 & 验证集：按真实 family 建边
# 🧩 对测试集：按 family 平均 embedding 推断再建边
# =========================================================
added_edges = []
for i in isolated_nodes:
    apk_i = apk_names[i]
    vec_i = vectors[i]
    ts_i = apk_timestamps_rel.get(apk_i, 0)

    # -----------------------------------------------
    # 训练集 & 验证集（标签已知） → 在同 family 找最相似节点
    # -----------------------------------------------
    if i in train_idx or i in val_idx:
        fam = apk_families.get(apk_i, -1)
        # members = [
        #     j for j in range(num_apk)
        #     if j != i and apk_families.get(apk_names[j], -1) == fam
        # ]
        members = [
            j for j in sorted(train_idx | val_idx)
            if j != i and apk_families.get(apk_names[j], -1) == fam
        ]
        if len(members) == 0:
            print(f"⚠ 节点 {apk_i} 的 family 内无其他节点，跳过")
            continue
        best_j = max(members, key=lambda j: sim_matrix[i, j])
        best_sim = sim_matrix[i, best_j]
        ts_j = apk_timestamps_rel.get(apk_names[best_j], 0)
        # mini TGN timestamp rule
        if ts_i <= ts_j:
            src, dst, edge_time = i, best_j, ts_j
        else:
            src, dst, edge_time = best_j, i, ts_i
        added_edges.append((src, dst, edge_time, fam, best_sim))
        print(f"Train/Val 节点 {apk_i} → 同 family {fam} 内连接 {apk_names[best_j]} (sim={best_sim:.4f})")
        continue

    # -----------------------------------------------
    # 测试集节点（标签未知） → 先找最近的 family 平均 embedding
    # -----------------------------------------------
    best_fam = None
    best_fam_sim = -1
    for fam_id, fam_avg in family_embeddings.items():
        # sim_fam = cosine_similarity([vec_i], [fam_avg])[0][0]
        # sim_fam = euclidean_distances([vec_i], [fam_avg])[0][0]
        sim_fam = euclidean_distances([vec_i], [fam_avg])[0][0]
        # print("sim_fam",sim_fam)
        sim_fam = 1 / (1 + sim_fam)
        # print("sim_fam", sim_fam)
        if sim_fam > best_fam_sim:
            best_fam_sim = sim_fam
            best_fam = fam_id
    # 在推断出的 family 里找最相似的真实节点（必须是 train+val）
    candidate_members = [
        j for j in range(num_apk)
        if (j in train_idx or j in val_idx)
        and apk_families.get(apk_names[j], -1) == best_fam
        and j != i
    ]
    if len(candidate_members) == 0:
        print(f"⚠ 推断 family {best_fam} 在 train/val 中无节点，测试节点 {apk_i} 无法连接")
        continue

    best_j = max(candidate_members, key=lambda j: sim_matrix[i, j])
    best_sim = sim_matrix[i, best_j]
    ts_j = apk_timestamps_rel.get(apk_names[best_j], 0)
    if ts_i <= ts_j:
        src, dst, edge_time = i, best_j, ts_j
        fam = apk_families.get(apk_names[best_j], -1)
    else:
        src, dst, edge_time = best_j, i, ts_i
        fam = apk_families.get(apk_names[i], -1)
    # added_edges.append((src, dst, edge_time, best_fam, best_sim))
    added_edges.append((src, dst, edge_time, fam, best_sim))
    print(f"Test 节点 {apk_i} → 推断 family {best_fam} 内连接 {apk_names[best_j]} (sim={best_sim:.4f})")
# 将新增边加到主边列表
edges.extend(added_edges)
print(f"✅ 最终补充孤立节点边数：{len(added_edges)}")

import csv
from collections import Counter
import numpy as np
# =========================================================
# 统计每个节点连接数量（节点度数）
# =========================================================
degree_count = Counter()
for i, j, _, _, _ in edges:
    degree_count[i] += 1
    degree_count[j] += 1
# 转成列表并按度数降序排列
sorted_degrees = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)

# =========================================================
# 输出统计信息
# =========================================================
degrees = np.array([deg for _, deg in sorted_degrees])
print("\n📈 节点连接统计：")
print(f"  总节点数: {num_apk}")
print(f"  有边的节点数: {len(degree_count)}")
print(f"  孤立节点数(度=0): {num_apk - len(degree_count)}")

# =========================================================
# 写入 CSV 文件
# =========================================================
# degree_csv = "./advanced_node_degree_stats_new_3.csv"
degree_csv = "./malradar_node_degree_stats.csv"
with open(degree_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "apk_name", "degree"])
    for node_id, deg in sorted_degrees:
        writer.writerow([node_id, apk_names[node_id], deg])
print(f"✅ 每个节点的连接数量（按降序）已保存到 {degree_csv}")

# =========================================================
# 5️⃣ 输出为 TGN 格式文件（包含 vector）
# =========================================================
# output_file = "./tgn_dataset_edges.txt"
# with open(output_file, "w") as f:
#     for edge_id, (i, j, ts, fam, sim) in enumerate(edges):
#     # for edge_id, (i, j, ts, sim) in enumerate(edges):
#         vec_i = " ".join(map(str, vectors[i].astype(int)))
#         vec_j = " ".join(map(str, vectors[j].astype(int)))
#         # f.write(f"{i} {j} {ts} {sim:.4f} {vec_i} {vec_j}\n")
#         # 🆕 family数字在ts后面
#         f.write(f"{i} {j} {ts} {fam} {sim:.4f} {vec_i} {vec_j}\n")
# print(f"✅ 图数据保存到: {output_file}")

import csv
# output_file = "advanced_tgn_dataset_edges_new_3.csv"
output_file = "./malradar_tgn_dataset_edges.csv"
# 写入 CSV 文件
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    # 写表头（可选）
    writer.writerow(["src", "dst", "timestamp", "family", "similarity", "vector_i", "vector_j"])
    # 写每一行
    for edge_id, (i, j, ts, fam, sim) in enumerate(edges):
        # vec_i = " ".join(map(str, vectors[i].astype(int)))
        # vec_j = " ".join(map(str, vectors[j].astype(int)))
        # writer.writerow([i, j, ts, fam, f"{sim:.4f}", vec_i, vec_j])
        vec_i_list = vectors[i].astype(int).tolist()
        vec_j_list = vectors[j].astype(int).tolist()
        writer.writerow([i, j, ts, fam, sim] + vec_i_list + vec_j_list)
print(f"✅ 图数据已保存为 CSV 文件: {output_file}")

# =========================================================
# 6️⃣ 同时保存节点索引映射
# =========================================================
# index_map_file = "advanced_apk_node_index_new_3.txt"
index_map_file = "./malradar_apk_node_index.txt"
with open(index_map_file, "w") as f:
    for idx, apk in enumerate(apk_names):
        ts = apk_timestamps_rel.get(apk, "1970-01-01")
        f.write(f"{idx}\t{apk}\t{ts}\n")
print(f"✅ 节点索引映射保存到: {index_map_file}")

# =========================================================
# 检测不同家族之间的边（按相似度排序后保存）
# =========================================================
# 存储跨家族的边信息：(src, dst, src_family, dst_family, sim, timestamp)
# 新增 timestamp 字段，避免后续查找，提升效率
cross_family_edges = []
# 统计跨家族的家族对（避免重复，用frozenset存储无序对）
family_pairs = set()

for src, dst, ts, _, sim in edges:
    # 获取两个节点的家族ID
    src_apk = apk_names[src]
    dst_apk = apk_names[dst]
    src_family = apk_families.get(src_apk, -1)  # -1表示无家族信息
    dst_family = apk_families.get(dst_apk, -1)

    # 排除有节点无家族信息的情况（可选，根据需求调整）
    if src_family == -1 or dst_family == -1:
        continue

    # 判断是否为不同家族
    if src_family != dst_family:
        # 直接存储 timestamp，无需后续循环查找
        cross_family_edges.append((src, dst, src_family, dst_family, sim, ts))
        # 记录家族对（无序，避免(a,b)和(b,a)重复）
        family_pair = frozenset([src_family, dst_family])
        family_pairs.add(family_pair)

# =========================================================
# 按相似度降序排序
# =========================================================
# 以列表中第4个元素（sim，相似度）为key，reverse=True表示降序
cross_family_edges_sorted = sorted(cross_family_edges, key=lambda x: x[4], reverse=True)

# 输出统计结果
print("\n🔍 不同家族之间的边统计（按相似度降序）：")
print(f"  跨家族的边总数：{len(cross_family_edges_sorted)}")
print(f"  涉及的不同家族对数量：{len(family_pairs)}")

# 输出前10条高相似度跨家族边的具体信息
if cross_family_edges_sorted:
    print("\n  前10条高相似度跨家族边示例：")
    for i, (src, dst, src_fam, dst_fam, sim, ts) in enumerate(cross_family_edges_sorted[:10]):
        src_apk = apk_names[src]
        dst_apk = apk_names[dst]
        print(f"  边 {i + 1}：APK {src_apk}（家族{src_fam}） <-> APK {dst_apk}（家族{dst_fam}），相似度：{sim:.4f}，时间戳：{ts}")

# =========================================================
# 保存排序后的跨家族边到CSV
# =========================================================
cross_family_csv = "./malradar_cross_family_edges_sorted.csv"
# cross_family_csv = "./advanced_cross_family_edges_sorted_3.csv"
with open(cross_family_csv, "w", newline="") as f:
    writer = csv.writer(f)
    # 表头字段与排序后的列表元素对应
    writer.writerow(
        ["src_node", "dst_node", "src_family", "dst_family", "similarity", "timestamp", "src_apk", "dst_apk"])
    for src, dst, src_fam, dst_fam, sim, ts in cross_family_edges_sorted:
        src_apk = apk_names[src]
        dst_apk = apk_names[dst]
        # 按表头顺序写入数据
        writer.writerow([src, dst, src_fam, dst_fam, sim, ts, src_apk, dst_apk])
print(f"\n✅ 按相似度降序排列的跨家族边信息已保存到：{cross_family_csv}")

# =========================================================
# 📌 统计测试节点的交互数量（不重复计数）
# =========================================================
test_interactions = 0
visited_edges = set()  # 防止重复计数
for (src, dst, ts, fam, sim) in edges:
    edge_id = tuple(sorted((src, dst)))  # 用无序对去重
    # 如果这条边已经统计过了，跳过
    if edge_id in visited_edges:
        continue
    # 检查是否有测试节点参与
    if src in test_idx or dst in test_idx:
        test_interactions += 1
        visited_edges.add(edge_id)
print(f"\n🔢 测试节点交互总数（去重后）: {test_interactions}")
print(f"📌 测试集节点数: {len(test_idx)}")
print(f"📌 总边数: {len(edges)}")
total_edges = len(edges)
if total_edges > 0:
    ratio = test_interactions / total_edges
    print(f"\n📊 测试节点交互占总边数比例: {ratio:.4f} ({ratio*100:.2f}%)")
    print("1-ratio",1-ratio)
else:
    print("❗ 总边数为 0，无法计算比例")
