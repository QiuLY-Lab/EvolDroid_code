import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # 将标量时间信息（timestamp）编码为一个固定维度的向量表示，以便模型能够感知时间间隔和时间顺序。
  # 它的核心思想是：使用一组不同频率的余弦函数，对时间进行周期性编码。
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1))
    # 手动初始化权重（关键部分）
    # np.linspace(0, 9, dimension)
    # → 生成 dimension 个数，在 [0, 9] 区间均匀分布
    # 10 ** np.linspace(0, 9, dimension)
    # → 得到不同数量级的时间尺度（从 10^0 到 10^9）
    # 1 / 10 ** ...
    # → 得到一组 指数递减的频率
    # 最终权重形状：[dimension, 1] 这等价于为每一维时间编码分配一个不同的“频率”。

    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())
    # 偏置项全部初始化为 0，不引入额外相位偏移

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output
