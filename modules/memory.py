import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):  # 备份内存状态
    # 创建当前内存状态的完整副本（深拷贝），用于后续恢复。
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    # 备份 self.memory.data：所有节点的当前内存向量（clone() 确保创建独立副本，不与原数据共享内存）。
    # 备份 self.last_update.data：所有节点内存的最后更新时间戳（同样深拷贝）。
    # 备份 self.messages：存储的待处理消息（键为节点 ID，值为消息列表），每条消息包含消息向量和对应的时间戳，均通过 clone() 深拷贝。
    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):  # 恢复内存状态
    # 将内存状态恢复到 backup_memory 备份时的状态。
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    # 分离内存的梯度
    # 切断内存相关变量与计算图的连接，阻止梯度从这些变量向后传播。

    # # 对 self.memory 调用 detach_()：移除内存向量的梯度跟踪（原地操作）。
    self.memory.detach_()

    # Detach all stored messages
    # # 对 self.messages 中的所有消息向量调用 detach()：消息的时间戳不需要梯度，仅分离消息向量。
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        # message[0] 是消息向量（张量） message[1] 是消息对应的时间戳（通常是张量或数值）
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages
      # 在时序模型中，通常不需要对历史内存状态计算梯度（仅关注当前更新），detach 可以减少不必要的梯度计算，节省内存。
      # 避免梯度从当前步骤回溯到过早期的内存状态，防止梯度爆炸或训练不稳定。

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
