import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
        # 主要用于设置和初始化与节点内存管理相关的一系列组件
        self.memory_dimension = memory_dimension  # 内存向量的维度
        self.memory_update_at_start = memory_update_at_start  # 标记内存更新时机（开始时/结束时）
        # 计算 "原始消息" 的维度，原始消息是节点间交互产生的基础信息，由四部分组成：
        # 源节点内存（memory_dimension）
        # 目标节点内存（memory_dimension）
        # 边特征（n_edge_features）
        # 时间编码（time_encoder.dimension，由时间编码器输出的维度）
        # 乘以 2，是因为 原始消息需要同时包含 “源节点内存” 和 “目标节点内存” 两部分信息，
        print("message_dimension111111",message_dimension)
        raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
        # 根据消息函数类型调整最终消息维度：
        # 如果消息函数是 "identity"（恒等函数，不改变消息结构），则消息维度等于原始消息维度。
        # 其他消息函数（如 MLP）会对原始消息进行处理，此时使用指定的message_dimension作为输出维度。
        message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
        print("message_dimension222222", message_dimension)
        # 管理所有节点的内存状态，包括每个节点的当前内存向量和最后更新时间。
        # n_nodes：节点总数
        # memory_dimension：内存向量维度
        # input_dimension和message_dimension：用于适配消息处理的维度
        self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
        # 通过工厂函数get_message_aggregator获取消息聚合器，用于将同一节点的多条消息合并为单条消息（例如按时间加权、取最后一条等策略）。
        # aggregator_type：指定聚合策略（如 "last" 取最新消息、"mean" 取平均等）。
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
        # 通过工厂函数get_message_function获取消息函数，用于对原始消息进行加工（例如用 MLP 处理、或直接返回原始消息）。
        # 参数说明： module_type：消息处理类型（如 "mlp"、"identity"） 输入 / 输出维度：匹配原始消息和目标消息的维度
        self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
        # 通过工厂函数get_memory_updater获取内存更新器，用于根据聚合后的消息更新节点的内存状态（例如用 GRU、MLP 等方式更新）。
        # 依赖已初始化的self.memory，并关联消息维度和内存维度。
        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
      # 定义了一个名为 compute_temporal_embeddings 的方法，用于计算时序图中源节点、目标节点和负采样节点的时序嵌入。它是时序图神经网络（Temporal GNN）中的核心方法，负责融合节点的历史交互、时间信息和动态记忆，为后续的任务（如链路预测、节点分类）提供输入特征
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
      # source_nodes：源节点 ID 列表（如交互中的 "用户"，形状 [batch_size]）
      # destination_nodes：目标节点 ID 列表（如交互中的 "物品"，形状 [batch_size]）
      # negative_nodes：负采样节点 ID 列表（用于训练的负例，形状 [batch_size]）
      # edge_times：交互发生的时间戳列表（形状 [batch_size]）
      # edge_idxs：交互的边索引（用于获取边特征，形状 [batch_size]）
      # n_neighbors：每层聚合时考虑的邻居数量

    n_samples = len(source_nodes)
    # # 合并所有需要计算嵌入的节点（源节点+目标节点+负采样节点）
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    # # 合并需要更新记忆的节点（源节点+目标节点，负例不更新记忆）
    positives = np.concatenate([source_nodes, destination_nodes])
    # # 合并对应的时间戳（三类节点共享同一批交互的时间戳）
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:  # 如果启用动态记忆
      if self.memory_update_at_start:
        # # 先更新记忆：使用之前批次存储的消息更新所有节点的记忆
        # Update memory for all nodes with messages stored in previous batches

        # 这里的 memory（临时变量）和 self.memory（正式模块）的关系是：
        # self.memory：是模型全局维护的 “正式记忆模块”，存储着所有节点的当前状态，对外提供 get_memory()（读取）、store_raw_messages()（存消息）等接口。
        # get_updated_memory(所有节点, 暂存消息)：是一个 “计算函数”—— 它接收所有节点的当前状态（从 self.memory 读取）和暂存的未处理消息（self.memory.messages），通过记忆更新逻辑（如 RNN、线性融合）计算出 “所有节点更新后的状态”，并返回两个结果：
        # memory：临时记忆变量，存储着 “所有节点更新后的状态”（是计算结果的 “副本”）；
        # last_update：所有节点的最新更新时间戳。
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        # # 直接获取当前记忆和最后更新时间
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      # # 计算时间差：当前交互时间与节点最后一次更新记忆的时间差
      # # 并进行标准化（减去均值，除以标准差）
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      # # 合并所有时间差
      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)

    # Compute the embeddings using the embedding module
    # # 通过嵌入模块（如GraphSumEmbedding、TemporalAttentionLayer等）计算所有节点的嵌入
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    # # 拆分嵌入结果：源节点、目标节点、负采样节点
    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    # 更新记忆（关键步骤）
    # 如果启用动态记忆，需要根据最新交互更新节点状态：
    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        # # 将之前的消息更新到记忆中（仅针对正例节点）
        # 把 get_updated_memory 函数计算出的 “正例节点更新结果”，写入到正式记忆模块 self.memory 中
        # （即让正式模块的状态与临时变量的状态同步）。
        self.update_memory(positives, self.memory.messages)

        # 记录每个epoch的误差
        # current_diff = torch.abs(memory[positives] - self.memory.get_memory(positives)).max().item()
        # print(f"Epoch max memory diff: {current_diff}")

        # memory[positives]：更新前，正例节点在 “临时记忆变量” 中的状态（代码前文通过 get_updated_memory 生成的临时记忆）。
        # self.memory.get_memory(positives)：更新后，从 “正式记忆模块” 中读取的正例节点状态。
        # torch.allclose(a, b, atol=1e-5)：判断两个张量 a 和 b 是否在 “绝对误差 1e-5” 范围内相等（允许微小的浮点误差）。
        # 若不相等，会抛出断言错误，提示 “记忆更新逻辑存在问题”（如消息未正确传递、参数计算错误等）
        # torch.allclose 中的 atol=1e-5（绝对误差容忍度）是必要的，原因是浮点计算的固有特性：
        # 在深度学习中，记忆更新涉及大量浮点运算（如矩阵乘法、激活函数计算），不同存储位置（临时变量、正式模块）的同一数值，可能因浮点精度损失（如 32 位浮点数的舍入误差）出现微小差异（例如 1.000001 和 1.000002）；
        # 这种差异是数学计算层面的正常现象，不代表逻辑错误，因此需要设置一个合理的误差容忍度，避免 “误报” 断言错误
        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-4), \
          "Something wrong in how the memory was updated"

        # # 清除已处理的消息（避免重复更新）
        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      # 根据节点间的交互（如用户 - 物品点击、社交网络好友互动），生成节点间的 “交互消息”，
      # 再将这些消息存入或用于更新节点的动态记忆（memory），让节点记住最新的交互信息。
      # # 生成源节点到目标节点的消息（基于它们的嵌入和交互信息）
      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      # # 生成目标节点到源节点的消息（反向消息，用于双向更新）
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      # # 存储或更新消息到记忆中
      if self.memory_update_at_start:
          # 情况1：先存储消息，等后续批次统一更新记忆
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
          # 情况2：立即用消息更新记忆（实时更新节点状态）
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      # # 如果使用dyrep机制（动态表示学习），直接使用记忆中的状态作为嵌入
      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
      # 链路预测的核心：基于节点的时序嵌入，计算 “源节点 - 目标节点”（正例）和 “源节点 - 负采样节点”（负例）发生交互的概率，用于模型训练（如通过正负概率差优化损失）。
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    # # 调用之前讲的compute_temporal_embeddings方法，得到三类节点的时序嵌入
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    # # 拼接源节点嵌入（正例用一次，负例再用一次）→ 形状 [2*n_samples, embedding_dim]
    # # 拼接正例目标嵌入和负例目标嵌入 → 形状 [2*n_samples, embedding_dim]
    # # 用affinity_score计算所有节点对的交互得分 → 形状 [2*n_samples, 1]

    # 为什么拼接源节点嵌入？# 因为每个源节点需要与 “正例目标” 和 “负例目标” 各计算一次得分（如用户 1 需要计算与物品 A（正）、物品 D（负）的得分），所以将源节点嵌入复制一份，与正 / 负目标嵌入拼接，批量计算效率更高。
    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    # # 拆分得分：前n_samples个是正例得分，后n_samples个是负例得分
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    # 将节点交互产生的 “消息” 先聚合（避免同一节点的消息重复处理），
    # 再通过消息函数优化消息表达，最后调用记忆更新器将聚合后的消息融合到节点的记忆（memory）中，
    # 实现节点状态的动态更新。
    # Aggregate messages for the same nodes
    # # 调用消息聚合器，对同一节点的多条消息进行聚合
    # 一个节点可能在一次批次中产生多条消息（例如 “用户 1” 在同一批次中点击了 “物品 A” 和 “物品 B”，
    # 会生成两条 “用户 1→物品” 的消息）。如果直接用多条消息更新同一节点的记忆，会导致节点状态被重复修改，
    # 效率低且可能引入噪声。因此需要先将 “同一节点的所有消息” 聚合为 “一条综合消息”，再用这条综合消息更新记忆。
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:  # # 确保有需要更新的节点
        # # 调用消息函数，对聚合后的消息进行非线性变换或特征优化
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    # 为 TGN（时序图网络）的内存更新模块生成 “原始消息”
    # 这些 raw messages 将在后续步骤中：
    # 按节点聚合（如 mean / last / attention），用于更新节点的 memory state
    # 一句话概括：将一次时序边交互转化为可用于节点内存更新的消息表示。

    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    # 获取源节点与目标节点的“状态表示”
    # 如果 use_source_embedding_in_message = False
    # 使用 memory 中存储的历史状态
    # 否则 使用当前计算得到的 source_node_embedding
    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)
    # self.memory.last_update[source_nodes]：源节点上一次被更新的时间
    # source_time_delta：当前交互距离上一次 memory 更新的时间间隔

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
