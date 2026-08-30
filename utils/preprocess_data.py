import json
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []
  e_feat = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      # e = line.strip().split(' ')
      # src,dst,timestamp,family,similarity,vector_i,vector_j
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[5:]])
      efeat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
      e_feat.append(efeat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l),np.array(e_feat)

def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    print("df.i.min()",df.i.min())
    print("df.u.min()",df.u.min())
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df

def run(data_name, bipartite=False):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  PATH = "/home/qly/mycode/ExtractAPIGraphFeature/ExtractAPIGraph/malradar_tgn_dataset_edges.csv"
  print("PATH",PATH)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat, efeat = preprocess(PATH)
  print("feat.shape[0]",feat.shape[0])
  print("feat.shape[1]",feat.shape[1])
  # feat.shape[0] 26470
  # feat.shape[1] 4000

  bipartite = False
  new_df = reindex(df, bipartite)

  # empty = np.zeros(feat.shape[1])[np.newaxis, :]  # 第一行新增空特征（索引0，占位用）
  # feat = np.vstack([empty, feat])

  eempty = np.zeros(efeat.shape[1])[np.newaxis, :]  # 第一行新增空特征（索引0，占位用）
  efeat = np.vstack([eempty, efeat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  # rand_feat = np.zeros((max_idx + 1, 172))
  # node_feat = feat[:, :feat.shape[1] // 2]

  # 拆成 src 和 dst
  src_feat = feat[:, :feat.shape[1] // 2]
  dst_feat = feat[:, feat.shape[1] // 2:]
  # 构造节点ID → 特征字典
  node_features_dict = {}
  for u, uf in zip(df['u'].values, src_feat):
    node_features_dict[u] = uf
  for i, if_ in zip(df['i'].values, dst_feat):
    node_features_dict[i] = if_
  # 按节点ID顺序堆叠
  max_node_id = max(max(df.u), max(df.i))
  node_features = np.zeros((max_node_id + 1, src_feat.shape[1]))
  for nid, feat_vec in node_features_dict.items():
    node_features[nid] = feat_vec

  new_df = new_df.sort_values('ts').reset_index(drop=True)
  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, efeat)  # # 交互特征
  # np.save(OUT_NODE_FEAT, rand_feat)  # 保存的是节点特征（node features），即每个节点（用户或物品）自身的特征。
  # np.save(OUT_NODE_FEAT, node_feat)
  print("efeat.shape", efeat.shape)
  print("node_feat.shape",node_features.shape)

  empty = np.zeros(node_features.shape[1])[np.newaxis, :]  # 第一行新增空特征（索引0，占位用）
  node_features = np.vstack([empty, node_features])
  print("node_features.shape",node_features.shape)

  np.save(OUT_NODE_FEAT, node_features)

  graph_df = pd.read_csv('./data/ml_{}.csv'.format(data_name))
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values

  # 1. 获取所有唯一值（可选，用于查看具体有哪些不同值）
  unique_labels = np.unique(labels)
  # 2. 统计唯一值的数量
  num_unique_labels = len(unique_labels)
  print(f"不同label的数量：{num_unique_labels}")
  print(f"所有不同label：{unique_labels}")
  # print("labels",labels[:10])
  # print("edge_idxs",edge_idxs[:10])
  # print("sources",sources[:10])
  # print("destinations",destinations[:10])
  timestamps = graph_df.ts.values
  # print("timestamps",timestamps[:10])

  # 1. 获取所有唯一值（可选，用于查看具体有哪些不同值）
  unique_time = np.unique(timestamps)
  # print("unique_时间戳有多少",unique_time)
  # 2. 统计唯一值的数量
  num_unique_time = len(unique_time)
  print(f"不同时间戳的数量：{num_unique_time}")

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
args = parser.parse_args()

run(args.data, bipartite=args.bipartite)