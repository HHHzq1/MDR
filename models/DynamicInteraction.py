"""
MDR_ads 动态交互层
单分支架构，使用3个Cell进行多模态特征交互
"""

import torch
import torch.nn as nn
import copy

from models.Cells import (
    GlobalLocalAlignmentCell, 
    GlobalEnhancedSemanticCell, 
    Reversed_GlobalLocalAlignmentCell
)


def unsqueeze2d(x):
    """在最后添加两个维度"""
    return x.unsqueeze(-1).unsqueeze(-1)


def unsqueeze3d(x):
    """在最后添加三个维度"""
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def clones(module, N):
    """产生N个相同的层"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicInteraction_Layer0(nn.Module):
    """
    动态交互第0层
    使用3个Cell进行初始特征交互：
    - GLAC: 文本域全局-局部对齐
    - GESC: 图像域全局增强语义
    - R_GLAC: 互补信息处理
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = 3  # 固定为3个Cell
        self.num_out_path = num_out_path
        
        # 三个核心Cell
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)  # 文本域
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)  # 图像域
        self.r_glac = Reversed_GlobalLocalAlignmentCell(args, num_out_path)  # 互补

    def forward(self, text, image):
        """
        Args:
            text: 文本特征 (bsz, seq_len, dim)
            image: 图像特征 (bsz, num_patches, dim)
        Returns:
            aggr_res_lst: 聚合后的特征列表
            all_path_prob: 路径概率 (bsz, num_out_path, num_cell)
        """
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        
        # 三个Cell的前向计算
        emb_lst[0], path_prob[0] = self.glac(text, image)  # 文本域
        emb_lst[1], path_prob[1] = self.gesc(text, image)  # 图像域
        emb_lst[2], path_prob[2] = self.r_glac(image, text)  # 互补

        # 门控掩码：当总路径概率过小时使用skip connection
        gate_mask = (sum(path_prob) < self.threshold).float()
        
        # 归一化路径概率
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        # 聚合各路径的输出
        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])
                cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class DynamicInteraction_Layer(nn.Module):
    """
    动态交互中间层
    基于前一层的输出继续进行特征交互
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = 3  # 固定为3个Cell
        self.num_out_path = num_out_path
        
        # 三个核心Cell
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)
        self.r_glac = Reversed_GlobalLocalAlignmentCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):
        """
        Args:
            ref_wrd: 前一层的输出特征列表
            text: 原始文本特征
            image: 原始图像特征
        Returns:
            aggr_res_lst: 聚合后的特征列表
            all_path_prob: 路径概率
        """
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        
        # 三个Cell的前向计算（使用前一层的输出）
        emb_lst[0], path_prob[0] = self.glac(ref_wrd[0], image)
        emb_lst[1], path_prob[1] = self.gesc(ref_wrd[1], image)
        emb_lst[2], path_prob[2] = self.r_glac(ref_wrd[2], text)
        
        if self.num_out_path == 1:
            # 单输出路径模式
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float()
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            # 多输出路径模式
            gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob