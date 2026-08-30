import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
# 不需要给参数赋值，只要在命令行中写上 --memory_update_at_end，就表示启用这个功能（值为 True）
# 如果不写这个参数，就表示不启用（值为 False）
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=2000, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


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
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
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

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
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
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        # timestamps_batch [3.060288e+08 3.089664e+08 3.067200e+08 3.060288e+08 3.060288e+08
        #  3.060288e+08 3.087936e+08 3.060288e+08 3.060288e+08 3.087936e+08
        #  3.089664e+08 3.067200e+08 3.060288e+08 3.060288e+08 3.067200e+08
        #  3.060288e+08 3.089664e+08 3.060288e+08 3.087936e+08 3.087936e+08
        #  3.089664e+08 3.089664e+08 3.060288e+08 3.060288e+08 3.067200e+08
        #  3.067200e+08 3.067200e+08 3.074976e+08 3.089664e+08 3.067200e+08
        #  3.060288e+08 3.060288e+08 3.060288e+08 3.060288e+08 4.136832e+08
        #  4.132512e+08 4.172256e+08 4.110912e+08 4.111776e+08 4.115232e+08
        #  4.096224e+08 4.125600e+08 4.131648e+08 4.104864e+08 4.101408e+08
        #  4.163616e+08 4.096224e+08 3.475008e+08 3.475008e+08 4.602528e+08
        #  4.595616e+08 4.599072e+08 4.599072e+08 4.595616e+08 4.588704e+08
        #  4.583520e+08 4.595616e+08 4.356288e+08 4.589568e+08 4.591296e+08
        #  4.599072e+08 4.604256e+08 4.574016e+08 4.590432e+08 4.593024e+08
        #  4.572288e+08 4.603392e+08 4.356288e+08 4.592160e+08 4.582656e+08
        #  4.589568e+08 4.603392e+08 4.586112e+08 4.580064e+08 4.571424e+08
        #  4.567968e+08 4.598208e+08 4.579200e+08 4.596480e+08 4.331232e+08
        #  4.586976e+08 4.331232e+08 4.331232e+08 4.591296e+08 4.592160e+08
        #  4.331232e+08 4.331232e+08 4.589568e+08 4.586112e+08 4.356288e+08
        #  4.577472e+08 4.603392e+08 4.589568e+08 4.579200e+08 4.215456e+08
        #  4.601664e+08 4.599072e+08 4.598208e+08 4.578336e+08 4.595616e+08
        #  4.586112e+08 4.581792e+08 4.591296e+08 4.585248e+08 4.601664e+08
        #  4.604256e+08 4.585248e+08 4.603392e+08 4.596480e+08 4.596480e+08
        #  4.598208e+08 4.586976e+08 4.596480e+08 4.331232e+08 4.571424e+08
        #  4.356288e+08 4.574880e+08 4.587840e+08 4.597344e+08 4.331232e+08
        #  4.356288e+08 4.565376e+08 4.599072e+08 4.599936e+08 4.604256e+08
        #  4.574016e+08 4.592160e+08 4.356288e+08 4.601664e+08 4.578336e+08
        #  4.582656e+08 4.586976e+08 4.589568e+08 4.596480e+08 4.593024e+08
        #  4.591296e+08 4.584384e+08 4.331232e+08 4.577472e+08 4.603392e+08
        #  4.572288e+08 4.603392e+08 4.600800e+08 4.577472e+08 4.573152e+08
        #  4.599072e+08 4.602528e+08 4.597344e+08 4.589568e+08 4.584384e+08
        #  4.571424e+08 4.597344e+08 4.568832e+08 4.356288e+08 4.331232e+08
        #  4.598208e+08 4.588704e+08 4.356288e+08 4.356288e+08 4.215456e+08
        #  4.567104e+08 4.577472e+08 4.590432e+08 4.331232e+08 4.597344e+08
        #  3.441312e+08 3.434400e+08 3.434400e+08 3.434400e+08 3.441312e+08
        #  3.441312e+08 3.441312e+08 3.434400e+08 3.441312e+08 3.441312e+08
        #  3.441312e+08 3.434400e+08 4.483296e+08 4.483296e+08 4.483296e+08
        #  3.867264e+08 4.435776e+08 4.438368e+08 4.435776e+08 4.435776e+08
        #  4.435776e+08 4.440960e+08 4.442688e+08 4.435776e+08 4.435776e+08
        #  4.435776e+08 4.435776e+08 4.435776e+08 4.435776e+08 4.440960e+08
        #  4.443552e+08 4.435776e+08 4.435776e+08 4.435776e+08 4.443552e+08]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
      # 时序图神经网络中内存（memory）的梯度截断操作，核心目的是避免梯度计算追溯到过远的历史，防止训练过程中梯度爆炸（gradient explosion）或计算效率过低。
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      # 在时序模型（如 TGN）中，节点的memory会随着每一批次的交互不断更新（通过消息传递和记忆更新机制）。
      # 由于神经网络训练需要计算梯度（反向传播），如果不做特殊处理，模型会默认从当前批次的损失出发，沿着记忆更新的链条，
      # 一直追溯计算到最开始的历史批次（“start of time”）。
      # 这会导致两个问题：
      # 梯度爆炸 / 消失：过长的梯度链条容易引发梯度爆炸（数值过大）或消失（数值趋近于 0），导致模型无法收敛。
      # 计算效率低下：存储和计算整个历史的梯度会占用大量内存和计算资源，拖慢训练速度。
      # detach_memory() 的作用tgn.memory.detach_memory() 是对节点记忆执行的梯度截断操作，它会：
      # 切断当前记忆状态与历史记忆状态之间的 “梯度连接”，使得反向传播时，梯度不会再追溯到 detach 操作之前的历史批次。
      # 保留当前的记忆状态值（不影响前向传播），但该状态被视为 “新的起点”，后续的梯度计算仅从这个点开始。
      # args.backprop_every 的意义代码注释中提到的 args.backprop_every 是一个超参数，
      # 用于控制 “每隔多少个批次执行一次 detach 操作”。例如：
      # 若 backprop_every=10，则每训练 10 个批次后，截断一次梯度，避免梯度链条超过 10 个批次的长度。
      # 这样既保证了一定的历史依赖被纳入梯度计算（10 个批次内），又避免了链条过长导致的问题。
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      # # 备份验证结束时的记忆状态（用于后续测试阶段恢复）
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      # # 恢复训练结束时的记忆状态（为接下来的“新节点验证”做准备）
      tgn.memory.restore_memory(train_memory_backup)
      # 时序模型的 memory 会随交互时间动态更新（包含节点的历史状态）。在常规验证（val_data）过程中，模型会基于验证数据更新记忆，但这会改变记忆状态。而接下来的 “新节点验证” 需要基于训练结束时的原始记忆状态（而非验证后被修改的状态），因为新节点验证的逻辑是：“模拟训练时未见过的节点突然出现，模型如何利用已有记忆进行预测”。

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
      # # 恢复到常规验证结束时的记忆状态（为下一轮训练或后续测试做准备）
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
      # # 2. 恢复验证后的记忆，准备新节点测试
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)  #  # 恢复验证后的记忆状态
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
