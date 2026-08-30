import numpy as np
import random
import pandas as pd
import os

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)

# ---------------------------
# 1️⃣ 解析 family & 时间戳
# ---------------------------
def parse_apk_family_time(txt_path, family_to_id):
    apk_family = {}

    with open(txt_path, "r") as f:
        # 跳过第一行（表头行）
        next(f, None)  # 使用next并设置默认值None，避免文件为空时报错
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                family, apk_id, date_str, time_str = line.split()
                family_id = family_to_id.get(family, -1)
                # 如果结果是 -1，输出 apk_id 和 family
                if family_id == -1:
                    print(f"id: {apk_id}, family: {family}")
                apk_family[apk_id.split(".")[0]] = family_to_id.get(family, -1)
            except ValueError:
                print(f"⚠️ 无法解析行：{line}")
                continue

    print(f"✅ 已提取 {len(apk_family)} 个APK记录")
    return apk_family

# ---------------------------
# 2️⃣ 解析 node_index 文件
# ---------------------------
def parse_node_index(index_file):
    """
    文件格式：
    0   b3f5f6...   1462810162.0
    """
    nodeid_to_apk = {}
    apk_to_nodeid = {}
    apk_timestamps = {}

    with open(index_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            node_id = int(parts[0])
            apk = parts[1]
            nodeid_to_apk[node_id] = apk
            apk_to_nodeid[apk] = node_id
            apk_timestamps[node_id] = parts[2]

    print(f"✅ 已读取 {len(nodeid_to_apk)} 个节点映射")
    return nodeid_to_apk, apk_to_nodeid,apk_timestamps

def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  # val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  # test_time = np.quantile(graph_df.ts, 0.8)
  # print("test_time",test_time)

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  # print("labels",labels[:10])
  # print("edge_idxs",edge_idxs[:10])
  # print("sources",sources[:10])
  # print("destinations",destinations[:10])
  timestamps = graph_df.ts.values
  # print("timestamps",timestamps[:10])

  random.seed(2020)

  # train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  # train_mask = timestamps <= test_time
  # test_mask = timestamps > test_time
  # val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  # 这里开始修改
  nodeid_to_apk, apk_to_nodeid, apk_timestamps_rel = parse_node_index("/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/advanced_apk_node_index_new_3.txt")
  # nodeid_to_apk, apk_to_nodeid, apk_timestamps_rel = parse_node_index("/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/malradar_family_ts_new.txt")
  # 1. 获取所有出现过的节点
  all_nodes = set(sources) | set(destinations)
  # 2. 为所有节点收集时间戳（未出现时间戳的节点过滤掉）
  nodes_with_ts = [(n, apk_timestamps_rel[n-1]) for n in all_nodes if n-1 in apk_timestamps_rel]
  # 3. 按 APK 时间戳排序
  nodes_sorted = sorted(nodes_with_ts, key=lambda x: x[1])
  # 4. 划分 0.8 训练节点
  split_idx = int(len(nodes_sorted) * 0.8) # CADE advanced
  # split_idx = int(len(nodes_sorted) * 0.85) # malradar
  train_nodes = set(n for n, _ in nodes_sorted[:split_idx])
  test_nodes = set(n for n, _ in nodes_sorted[split_idx:])
  print("get_data_node_classification！！！")
  print("训练节点数量:", len(train_nodes))
  print("测试节点数量:", len(test_nodes))
  print("get_data_node_classification！！！")
  # 5. 按节点划分边
  train_mask = []
  test_mask = []
  # for u, v in zip(sources, destinations):
  #     if (u in train_nodes) or (v in train_nodes):
  #         train_mask.append(True)
  #         test_mask.append(False)
  #     else:
  #         train_mask.append(False)
  #         test_mask.append(True)  # 只要边涉及测试节点 → 测试边

  for u, v in zip(sources, destinations):
      if (v in test_nodes):
          train_mask.append(False)
          test_mask.append(True)  # 只要边涉及测试节点 → 测试边
      else:
          train_mask.append(True)
          test_mask.append(False)

  train_mask = np.array(train_mask)
  test_mask = np.array(test_mask)
  ###

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  train_data = Data(sources[train_mask],
                    destinations[train_mask],
                    timestamps[train_mask],
                    edge_idxs[train_mask],
                    labels[train_mask])
  # val_data = Data(sources[val_mask],
  #                 destinations[val_mask],
  #                 timestamps[val_mask],
  #                 edge_idxs[val_mask],
  #                 labels[val_mask])
  test_data = Data(sources[test_mask],
                   destinations[test_mask],
                   timestamps[test_mask],
                   edge_idxs[test_mask],
                   labels[test_mask])

  # 🧮 统计每个数据集的标签分布
  def print_label_stats(name, lbls):
      from collections import Counter
      c = Counter(lbls)
      print(f"\n📊 {name} 集标签分布（类别: 样本数）:")
      for k, v in sorted(c.items()):
          print(f"  class {k:>3}: {v}")
      print(f"  总计: {len(lbls)} 样本")

  # print("get_data_node_classification！！！")
  # print_label_stats("训练集", train_data.labels)
  # print_label_stats("验证集", val_data.labels)
  # print_label_stats("测试集", test_data.labels)
  # print("get_data_node_classification！！！")
  # return full_data, node_features, edge_features, train_data, val_data, test_data
  return full_data, node_features, edge_features, train_data, test_data

def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))

  # # ✅ 按时间排序
  graph_df = graph_df.sort_values('ts').reset_index(drop=True)
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85])) # malradar

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))

  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  # new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
  print("n_total_unique_nodes",n_total_unique_nodes)
  print("len(test_node_set)",len(test_node_set))
  new_test_node_set = set(random.sample(test_node_set, int(0.05 * n_total_unique_nodes)))
  # n_total_unique_nodes 3719
  # len(test_node_set) 272

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  # train_data = Data(sources[train_mask][np.argsort(timestamps[train_mask])],
  #                   destinations[train_mask][np.argsort(timestamps[train_mask])],
  #                   timestamps[train_mask][np.argsort(timestamps[train_mask])],
  #                   edge_idxs[train_mask][np.argsort(timestamps[train_mask])],
  #                   labels[train_mask][np.argsort(timestamps[train_mask])])
  train_data = Data(sources[train_mask],
                    destinations[train_mask],
                    timestamps[train_mask],
                    edge_idxs[train_mask],
                    labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)  # 筛选出同时满足两个条件的边
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
  else:
    print("different_new_nodes_between_val_and_test",different_new_nodes_between_val_and_test)
    print("FalseFalseFalseFalseFalseFalse")
    edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask],
                  destinations[val_mask],
                  timestamps[val_mask],
                  edge_idxs[val_mask],
                  labels[val_mask])
  test_data = Data(sources[test_mask],
                   destinations[test_mask],
                   timestamps[test_mask],
                   edge_idxs[test_mask],
                   labels[test_mask])
  # val_data = Data(sources[val_mask][np.argsort(timestamps[val_mask])],
  #                 destinations[val_mask][np.argsort(timestamps[val_mask])],
  #                 timestamps[val_mask][np.argsort(timestamps[val_mask])],
  #                 edge_idxs[val_mask][np.argsort(timestamps[val_mask])],
  #                 labels[val_mask][np.argsort(timestamps[val_mask])])
  # test_data = Data(sources[test_mask][np.argsort(timestamps[test_mask])],
  #                  destinations[test_mask][np.argsort(timestamps[test_mask])],
  #                  timestamps[test_mask][np.argsort(timestamps[test_mask])],
  #                  edge_idxs[test_mask][np.argsort(timestamps[test_mask])],
  #                  labels[test_mask][np.argsort(timestamps[test_mask])])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask],
                           destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask],
                           labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask],
                            destinations[new_node_test_mask],
                            timestamps[new_node_test_mask],
                            edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])
  # new_node_val_data = Data(sources[new_node_val_mask][np.argsort(timestamps[new_node_val_mask])],
  #                          destinations[new_node_val_mask][np.argsort(timestamps[new_node_val_mask])],
  #                          timestamps[new_node_val_mask][np.argsort(timestamps[new_node_val_mask])],
  #                          edge_idxs[new_node_val_mask][np.argsort(timestamps[new_node_val_mask])],
  #                          labels[new_node_val_mask][np.argsort(timestamps[new_node_val_mask])])
  #
  # new_node_test_data = Data(sources[new_node_test_mask][np.argsort(timestamps[new_node_test_mask])],
  #                           destinations[new_node_test_mask][np.argsort(timestamps[new_node_test_mask])],
  #                           timestamps[new_node_test_mask][np.argsort(timestamps[new_node_test_mask])],
  #                           edge_idxs[new_node_test_mask][np.argsort(timestamps[new_node_test_mask])],
  #                           labels[new_node_test_mask][np.argsort(timestamps[new_node_test_mask])])
  print("训练集最后时间:", timestamps[train_mask].max())
  print("测试集最早时间:", timestamps[test_mask].min())
  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
