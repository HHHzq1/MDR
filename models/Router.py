"""
Router Module - Dynamic Routing Decision Maker
Used to compute path selection probabilities for each Cell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activateFunc(x):
    """Custom activation function: tanh + relu"""
    x = torch.tanh(x)
    return F.relu(x)


class Router(nn.Module):
    """
    Router module
    Computes probabilities for which output paths features should flow to via MLP network
    """
    def __init__(self, num_out_path, embed_size, hid):
        """
        Args:
            num_out_path: Number of output paths
            embed_size: Input feature dimension
            hid: Hidden layer dimension
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
        """Initialize bias to 1.5 to make path probabilities relatively balanced initially"""
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        """
        Args:
            x: Input features (bsz, seq_len, embed_size)
        Returns:
            Path probabilities (bsz, num_out_path)
        """
        x = x.mean(-2)  # Global average pooling
        x = self.mlp(x)
        soft_g = activateFunc(x) 
        return soft_g
