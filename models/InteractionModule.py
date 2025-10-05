"""
MDR_ads 交互模块
单分支动态路由模块，使用3个Cell进行多模态特征交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer


class MultimodalInteractionModule(nn.Module):
    """
    多模态交互模块
    通过多层动态路由实现文本-图像特征的深度交互
    - Layer0: 初始交互层
    - Layer1-N: 中间交互层（可多层）
    - Layer_final: 最终聚合层
    """
    def __init__(self, args, num_layer_routing=3, num_cells=3, path_hid=128):
        """
        Args:
            args: 参数配置
            num_layer_routing: 路由层数（默认3层）
            num_cells: Cell数量（固定为3）
            path_hid: 路径隐藏层维度
        """
        super(MultimodalInteractionModule, self).__init__()
        self.args = args
        self.num_cells = 3  # 固定为3个Cell：GLAC, GESC, R_GLAC
        
        # 第0层：初始交互（3个Cell -> 3个输出路径）
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(args, self.num_cells, self.num_cells)
        
        # 中间层：继续交互（3个Cell -> 3个输出路径）
        self.dynamic_itr_l1 = nn.ModuleList([
            DynamicInteraction_Layer(args, self.num_cells, self.num_cells) 
            for i in range(num_layer_routing - 2)
        ])
        
        # 最终层：聚合到单一输出（3个Cell -> 1个输出路径）
        self.dynamic_itr_l2 = DynamicInteraction_Layer(args, self.num_cells, 1)
        
        # 路径映射层（用于相似度计算）
        total_paths = self.num_cells ** 2 * (num_layer_routing - 1) + self.num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)

    def forward(self, text, image):
        """
        Args:
            text: 文本特征 (bsz, seq_len, dim)
            image: 图像特征 (bsz, num_patches, dim)
        Returns:
            pairs_emb_lst: 交互后的特征列表
            sim_paths: 路径相似度矩阵 (bsz, bsz)
        """
        mid_paths = []
        
        # 第0层：初始交互
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(text, image)
        
        # 中间层：逐层交互
        for module in self.dynamic_itr_l1:
            pairs_emb_lst, paths_l1 = module(pairs_emb_lst, text, image)
            mid_paths.append(paths_l1)
            
        # 最终层：聚合
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, text, image)
        
        n_img, n_stc = paths_l2.size()[:2]
        
        # 整理路径概率
        paths_l0 = paths_l0.view(n_img, n_stc, -1)
        
        for i in range(len(mid_paths)):
            if i == 0:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)
                paths_l1 = mid_paths[i]
            else:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)
                paths_l1 = torch.cat([paths_l1, mid_paths[i]], dim=-1)
                
        paths_l2 = paths_l2.view(n_img, n_stc, -1)
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1)
        
        # 计算路径相似度矩阵
        sim_paths = torch.matmul(paths.squeeze(-2), paths.squeeze(-2).transpose(-1, -2))
        
        return pairs_emb_lst, sim_paths