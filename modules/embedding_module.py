import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    # 实现基于时间的嵌入（embedding）生成机制
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
          # 与默认初始化不同，这里使用均值为 0、标准差为 1/sqrt(输入维度) 的正态分布初始化权重和偏置，这是一种更稳定的初始化方式
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)
    # 创建一个自定义的线性层 NormalLinear，输入维度为 1（时间差通常是标量），输出维度为 n_node_features（节点特征维度）。
    # 这个层的作用是将时间差（标量）映射到与节点特征同维度的向量，以便后续与节点嵌入融合。

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    # 生成嵌入的核心逻辑，作用是将时间信息融合到节点的记忆（memory）中
    # 参数说明：
    # memory：存储节点当前状态的记忆体（可理解为节点的基础嵌入）。
    # source_nodes：需要生成嵌入的源节点 ID。
    # time_diffs：时间差（例如，当前时间与节点上一次交互时间的差值）。
    # 计算逻辑：
    # memory[source_nodes, :]：获取源节点在记忆体中的基础嵌入。
    # time_diffs.unsqueeze(1)：将时间差从一维向量（形状 [N]）扩展为二维（形状 [N, 1]），以匹配线性层的输入要求。
    # self.embedding_layer(...)：将时间差映射为与节点特征同维度的向量。
    # 最终嵌入 = 基础嵌入 × (1 + 时间映射向量)：通过乘法将时间信息融合到基础嵌入中，实现时间对节点嵌入的调制。
    # (1 + 时间映射向量) 避免时间向量为 0 时的失效问题
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
    # 用于实现基于图结构的嵌入（embedding）生成机制。这类机制主要用于从图数据（尤其是时序图）中提取节点的特征表示，融合了节点自身属性、邻居信息和时间因素
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  # compute_embedding 用于递归地计算节点在给定时间点的时序图嵌入（temporal embedding）。
    # 它通过堆叠 n_layers 层时序图注意力/卷积层，逐层聚合历史邻居的信息。
    # 核心思想是：
    # 每一层：
    # 先得到当前节点在上一层的表示
    # 再获取其时间上早于 timestamps 的邻居
    # 递归计算邻居的表示
    # 使用时间编码、边特征和注意力机制进行聚合
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    # 查询节点（source node）与自身的时间差为 0
    # 使用 time encoder 对 Δt = 0 进行编码
    # 用作 attention / aggregation 中的 query 时间特
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
        # 将 动态记忆（memory） 与静态特征相加
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:
      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors)

      # 获取时序邻居
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      # 计算 节点查询时间 − 边发生时间  用于时间编码（temporal encoding）
      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      # 对所有邻居节点 递归调用自身
      # 邻居使用与 source node 相同的查询时间 （不管是 source node，还是它的一阶、二阶、三阶邻居，
      # 都在同一个“查询时间点”上计算表示）
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      # 形状变为 [batch_size, n_neighbors, embedding_dim]
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      # ID 为 0 的邻居通常是 padding
      # mask 用于注意力计算中屏蔽无效项
      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)
      # 时序注意力（Temporal Attention）
      # 使用：
      # source node embedding
      # neighbor embedding
      # 时间编码
      # 边特征
      # 输出当前层的节点表示

      return source_embedding

  # 这是一个抽象接口，子类实现具体机制
  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    # 一种基于 “求和聚合” 的图嵌入模块。核心特点是通过 “对邻居特征求和” 的方式融合多层邻居信息，
    # 同时结合节点自身特征和时间信息，生成包含图结构与时序依赖的节点嵌入。
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    # 针对每一层聚合（n_layers 层），分别定义两个线性层：
    # linear_1：将邻居的 “嵌入 + 边时间特征 + 边特征” 融合并压缩到目标嵌入维度。
    # linear_2：将 “邻居求和结果” 与 “源节点自身特征 + 源节点时间特征” 融合并压缩到目标维度。
    # 使用 ModuleList 是为了支持多层聚合时，每层使用独立的可训练参数。
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


