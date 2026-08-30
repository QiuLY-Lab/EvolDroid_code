import math
import logging
import time
import sys
import random
import argparse
from pathlib import Path
import torch
import numpy as np
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import torch
from collections import defaultdict
import csv
import os
from tqdm import tqdm

@torch.no_grad()
def generate_destination_avg_embeddings(tgn, full_data, batch_size, num_neighbors, save_path,node_features):
    """
    ⚙️ 计算所有 destination 的平均 embedding，并保存到文件中（按时间戳排序）
    :param tgn: 训练好的 TGN 模型
    :param full_data: 包含所有边的数据（Data对象）
    :param batch_size: 每次处理的边数量
    :param num_neighbors: TGN采样邻居数
    :param save_path: 输出 CSV 文件路径
    """

    print("\n🚀 开始计算所有 destination 的平均 embedding...")
    device = next(tgn.parameters()).device  # 获取模型中第一个参数所在的设备（CPU 或 GPU）。

    num_instances = len(full_data.sources)
    num_batches = (num_instances + batch_size - 1) // batch_size

    # 用于存储每个 destination 的 embedding 列表
    dest_embs_dict = defaultdict(list)
    dest_label_dict = {}
    dest_time_dict = {}

    # ⚠️ 确保按时间戳顺序遍历，防止 memory 时间倒流
    sorted_idx = np.argsort(full_data.timestamps)
    sources_sorted = full_data.sources[sorted_idx]
    destinations_sorted = full_data.destinations[sorted_idx]
    timestamps_sorted = full_data.timestamps[sorted_idx]
    edge_idxs_sorted = full_data.edge_idxs[sorted_idx]
    labels_sorted = full_data.labels[sorted_idx]

    tgn = tgn.eval()
    for k in tqdm(range(num_batches), desc="Embedding计算中"):
        s_idx = k * batch_size
        e_idx = min(num_instances, s_idx + batch_size)

        sources_batch = torch.from_numpy(sources_sorted[s_idx:e_idx]).long().to(device)
        destinations_batch = torch.from_numpy(destinations_sorted[s_idx:e_idx]).long().to(device)
        timestamps_batch = torch.from_numpy(timestamps_sorted[s_idx:e_idx]).float().to(device)
        edge_idxs_batch = torch.from_numpy(edge_idxs_sorted[s_idx:e_idx]).long().to(device)

        # ✅ 通过TGN获取时序embedding
        # src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(
        #     sources_batch,
        #     destinations_batch,
        #     destinations_batch,
        #     timestamps_batch,
        #     edge_idxs_batch,
        #     num_neighbors
        # )
        src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(
            sources_batch.cpu().numpy(),
            destinations_batch.cpu().numpy(),
            destinations_batch.cpu().numpy(),
            timestamps_batch.cpu().numpy(),
            edge_idxs_batch.cpu().numpy(),
            num_neighbors
        )

        # 用于记录每个节点所有出现的时间与标签
        dst_history = defaultdict(list)

        # ✅ 保存每个destination的embedding
        for i, dst in enumerate(destinations_batch.cpu().numpy()):
            dest_embs_dict[dst].append(dst_emb[i].cpu().numpy())
            # 保存最新时间戳 & 标签（同节点多次出现时取最后一次）
            dest_label_dict[dst] = int(labels_sorted[s_idx + i])
            dest_time_dict[dst] = float(timestamps_sorted[s_idx + i])

            label = int(labels_sorted[s_idx + i])
            ts = float(timestamps_sorted[s_idx + i])
            dst_history[dst].append((ts, label))

        # 循环结束后分析
        # print("\n🕒 检查每个 destination 节点的时间/标签变化趋势：")
        for dst, records in dst_history.items():
            records = sorted(records, key=lambda x: x[0])  # 按时间排序
            times, labels = zip(*records)
            if list(times) != sorted(times):
                print(f"⚠️ 节点 {dst} 时间顺序异常：{times}")
            if len(set(labels)) > 1:
                print(f"ℹ️ 节点 {dst} 标签发生变化：{labels}")
    print(f"✅ 已提取 {len(dest_embs_dict)} 个节点的embedding")

    # 计算平均embedding
    avg_embs = []
    for dst, embs in dest_embs_dict.items():
        avg_embs.append({
            "node_id": dst,
            "timestamp": dest_time_dict[dst],
            "label": dest_label_dict[dst],
            "embedding": np.mean(np.stack(embs), axis=0),
            "raw_feature": node_features[dst].tolist(),
        })

    # 按时间排序保存
    avg_embs = sorted(avg_embs, key=lambda x: x["timestamp"])
    # 输出到CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    emb_dim = len(avg_embs[0]["embedding"])
    raw_dim = len(avg_embs[0]["raw_feature"])
    header = ["node_id", "timestamp", "label"] + [f"emb_{i}" for i in range(emb_dim)] + [f"raw_{i}" for i in range(raw_dim)]

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in avg_embs:
            row = [item["node_id"], item["timestamp"], item["label"]] + item["embedding"].tolist() + item["raw_feature"]
            writer.writerow(row)
    print(f"💾 已保存平均embedding文件: {save_path}")

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
parser.add_argument('--bs', type=int, default=8, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate') # 3e-4
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability') # 0.1
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true', help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=2000, help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + 'node-classification.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + 'node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

full_data, node_features, edge_features, train_data, val_data, test_data = get_data_node_classification(DATA, use_validation=args.use_validation)
max_idx = max(full_data.unique_nodes)
train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,i) if i > 0 else "results/{}_node_classification.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  save_path = "./data/avg_destination_embeddings.csv"
  generate_destination_avg_embeddings(
      tgn=tgn,
      full_data=full_data,
      batch_size=1024,
      num_neighbors=NUM_NEIGHBORS,
      save_path=save_path,
      node_features = node_features
  )
  print("已保存！！！！！！")
  # logger.debug('Num of training instances: {}'.format(num_instance))
  # logger.debug('Num of batches per epoch: {}'.format(num_batch))

  logger.info('Loading saved TGN model')
  model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
  tgn.load_state_dict(torch.load(model_path))
  tgn.eval()
  logger.info('TGN models loaded')
  logger.info('Start training node classification task')

  decoder = MLP(node_features.shape[1], drop=DROP_OUT)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)
  # decoder_loss_criterion = torch.nn.BCELoss()
  decoder_loss_criterion = torch.nn.CrossEntropyLoss()  # 多分类 用 CrossEntropyLoss

  train_losses = []
  import pandas as pd
  import torch
  from torch.utils.data import TensorDataset, DataLoader
  # === 1. 加载保存的 destination 平均 embedding 文件 ===
  df = pd.read_csv("./data/avg_destination_embeddings.csv")  # CSV 中包含这些列：'node_id', 'timestamp', 'label', 'emb_0', 'emb_1', ..., 'emb_N'
  print(f"📂 加载到 {len(df)} 条节点数据, 列: {df.columns[:5]} ...")

  # === 2. 按时间戳排序（确保是时间顺序） ===
  df = df.sort_values(by="timestamp").reset_index(drop=True)
  # === 3. 提取特征与标签 ===
  embedding_cols = [c for c in df.columns if c.startswith("emb_")]
  raw_cols = [c for c in df.columns if c.startswith("raw_")]

  X = torch.tensor(df[embedding_cols].values, dtype=torch.float32)
  # X = torch.tensor(df[raw_cols].values, dtype=torch.float32)
  # X = torch.tensor(df[embedding_cols + raw_cols].values, dtype=torch.float32)
  y = torch.tensor(df["label"].values, dtype=torch.long)

  # === 4. 按比例划分数据集（70% / 15% / 15%） ===
  num_samples = len(df)
  train_size = int(0.8 * num_samples)
  val_size = int(0.1 * num_samples)
  test_size = num_samples - train_size - val_size
  train_X, val_X, test_X = torch.split(X, [train_size, val_size, test_size])
  train_y, val_y, test_y = torch.split(y, [train_size, val_size, test_size])
  print(f"📊 训练集: {len(train_X)} | 验证集: {len(val_X)} | 测试集: {len(test_X)}")

  # 过采样开始
  from imblearn.over_sampling import RandomOverSampler
  # ⚠️ 如果 train_X / train_y 是 torch.Tensor，则先转成 numpy
  train_X_np = train_X.cpu().numpy() if isinstance(train_X, torch.Tensor) else train_X
  train_y_np = train_y.cpu().numpy() if isinstance(train_y, torch.Tensor) else train_y

  # # === 1️⃣ 随机过采样 (简单随机复制少数类样本)
  ros = RandomOverSampler(random_state=42)
  train_X_res, train_y_res = ros.fit_resample(train_X_np, train_y_np)

  # # === 2️⃣ 或使用 SMOTE (合成新样本，适用于连续特征)
  # smote = SMOTE(random_state=42, k_neighbors=5)
  # train_X_res, train_y_res = smote.fit_resample(train_X_np, train_y_np)

  # 打印过采样前后类别分布
  from collections import Counter
  print("📊 过采样前类别分布:", Counter(train_y_np))
  print("📈 过采样后类别分布:", Counter(train_y_res))

  # === 3️⃣ 转回 torch.Tensor（如果后续还要用 PyTorch 训练）
  train_X = torch.tensor(train_X_res, dtype=torch.float32)
  train_y = torch.tensor(train_y_res, dtype=torch.long)
  #
  # # 🧮 统计函数
  def print_label_stats(name, y_tensor):
      y_np = y_tensor.cpu().numpy()
      unique, counts = np.unique(y_np, return_counts=True)
      print(f"\n{name} 类别分布:")
      for u, c in zip(unique, counts):
          print(f"  类别 {u:>3}: {c}")
      print(f"  总计: {len(y_np)}")
  #
  # # 🚀 打印三个集合的类别分布
  print_label_stats("训练集", train_y)
  print_label_stats("验证集", val_y)
  print_label_stats("测试集", test_y)
  # # 过采样结束

  # === 5. 构建 DataLoader ===
  train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False)
  test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=False)

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    decoder.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        decoder_optimizer.zero_grad()
        logits = decoder(xb)
        loss = decoder_loss_criterion(logits, yb)
        loss.backward()
        decoder_optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # === 验证 ===
    decoder.eval()
    correct, total = 0, 0
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = decoder(xb)
            loss = decoder_loss_criterion(logits, yb)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
            correct += (preds == yb).sum().item()
            total += len(yb)
    val_acc = correct / total
    print(f"Epoch {epoch + 1:03d}: train_loss={avg_loss:.4f}, val_loss={val_loss / len(val_loader):.4f}, val_acc={val_acc:.4f}")
    if early_stopper.early_stop_check(val_acc):
        print(f"⛔ Early stopping at epoch {epoch + 1}")
        break

  from sklearn.metrics import f1_score, accuracy_score
  decoder.eval()
  all_preds, all_labels = [], []
  with torch.no_grad():
      for xb, yb in test_loader:
          xb, yb = xb.to(device), yb.to(device)
          preds = decoder(xb).argmax(dim=1)
          all_preds.append(preds.cpu())
          all_labels.append(yb.cpu())
          # print("preds.cpu()",preds.cpu())
          # print("yb.cpu()",yb.cpu())

  y_true = torch.cat(all_labels).numpy()
  y_pred = torch.cat(all_preds).numpy()

  test_acc = accuracy_score(y_true, y_pred)
  test_f1 = f1_score(y_true, y_pred, average="macro")
  test_f11 = f1_score(y_true, y_pred, average=None)

  from sklearn.metrics import recall_score
  import numpy as np
  # === 计算各类 recall ===
  recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
  # === 计算 G-Mean ===
  gmean = np.prod(recalls) ** (1 / len(recalls))
  print(f"✅ 测试集准确率: {test_acc:.4f}, F1-macro: {test_f1:.4f}, G-Mean: {gmean:.4f}")
  print("各类别 F1:", test_f11)
  print("各类别 Recall:", recalls)
