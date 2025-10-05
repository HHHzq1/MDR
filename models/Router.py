"""
Router模块 - 动态路由决策器
用于计算每个Cell的路径选择概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activateFunc(x):
    """自定义激活函数：tanh + relu"""
    x = torch.tanh(x)
    return F.relu(x)


class Router(nn.Module):
    """
    路由器模块
    通过MLP网络计算特征应该流向哪些输出路径的概率
    """
    def __init__(self, num_out_path, embed_size, hid):
        """
        Args:
            num_out_path: 输出路径数量
            embed_size: 输入特征维度
            hid: 隐藏层维度
        """
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hid), 
            nn.ReLU(True), 
            nn.Linear(hid, num_out_path)
        )
        self.init_weights()

    def init_weights(self):
        """初始化偏置为1.5，使路径概率初始时较为均衡"""
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (bsz, seq_len, embed_size)
        Returns:
            路径概率 (bsz, num_out_path)
        """
        x = x.mean(-2)  # 全局平均池化
        x = self.mlp(x)
        soft_g = activateFunc(x) 
        return soft_g
