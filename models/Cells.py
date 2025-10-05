"""
MDR_ads Cell模块定义
包含三个核心Cell：
- GlobalLocalAlignmentCell (GLAC): 文本域处理
- GlobalEnhancedSemanticCell (GESC): 图像域处理
- Reversed_GlobalLocalAlignmentCell (R_GLAC): 互补信息处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Router import Router
from transformers import BertConfig, CLIPConfig
from models.XModules import CrossModalAlignment, AttentionFiltration


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class BertPooler(nn.Module):
    """从BERT隐藏状态中提取[CLS] token的pooler"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 提取第一个token ([CLS])的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GlobalLocalAlignmentCell(nn.Module):
    """
    全局-局部对齐Cell (GLAC) - 文本域处理
    通过计算文本和图像的全局与局部相似性来增强特征表示
    """
    def __init__(self, args, num_out_path):
        super(GlobalLocalAlignmentCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name), args)
        self.SAF_module = AttentionFiltration(BertConfig.from_pretrained(args.bert_name).hidden_size)
        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(CLIPConfig.from_pretrained(args.vit_name).vision_config)
        self.fc_sim_tranloc = nn.Linear(768, 768)
        self.fc_sim_tranglo = nn.Linear(768, 768)
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, text, image):
        """计算文本-图像的全局和局部对齐"""
        # 局部相似性表征
        text_aware_image, _ = self.CrossModalAlignment(text, image)  # (bsz, seq_len, 768)

        sim_local = torch.pow(torch.sub(text, text_aware_image), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # 全局相似性表征
        text_cls_output = self.text_cls_pool(text)
        image_cls_output = self.image_cls_pool(image)
        sim_global = torch.pow(torch.sub(text_cls_output, image_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # 拼接全局和局部对齐
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)  # (bsz, seq_len+1, 768)

        # 相似图推理
        sim_emb = self.SAF_module(sim_emb)  # (bsz, 768)

        return sim_emb

    def forward(self, text, image):
        path_prob = self.router(text)
        
        sim_emb = self.alignment(text, image)
        sim_emb = sim_emb.unsqueeze(-2).expand(-1, text.size(1), -1)

        return sim_emb, path_prob


class GlobalEnhancedSemanticCell(nn.Module):
    """
    全局增强语义Cell (GESC) - 图像域处理
    通过门控机制融合文本和图像的全局语义信息
    """
    def __init__(self, args, num_out_path):
        super(GlobalEnhancedSemanticCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))

        self.fc_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 768)
        )

    def global_gate_fusion(self, text, image):
        """全局门控融合"""
        text_cls = self.text_cls_pool(text)  # (bsz, 768)
        image_cls = self.image_cls_pool(image)  # (bsz, 768)

        # 门控机制：全局信息对齐、融合
        gate_all = self.fc_mlp(text_cls + image_cls)  # (bsz, 768)
        gate = torch.softmax(gate_all, dim=-1)  # (bsz, 768)
        gate_out = gate * text_cls + (1 - gate) * image_cls  # (bsz, 768)
        gate_out = gate_out.unsqueeze(-2).expand(-1, text.size(1), -1)

        return gate_out

    def forward(self, text, image):
        path_prob = self.router(text)
        gate_out = self.global_gate_fusion(text, image)

        return gate_out, path_prob


class Reversed_GlobalLocalAlignmentCell(nn.Module):
    """
    反向全局-局部对齐Cell (R_GLAC) - 互补信息处理
    与GLAC相反，以图像为查询，文本为键值进行对齐
    """
    def __init__(self, args, num_out_path):
        super(Reversed_GlobalLocalAlignmentCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.CrossModalAlignment = CrossModalAlignment(CLIPConfig.from_pretrained(args.vit_name).vision_config, args)
        self.SAF_module = AttentionFiltration(CLIPConfig.from_pretrained(args.vit_name).vision_config.hidden_size)
        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(CLIPConfig.from_pretrained(args.vit_name).vision_config)
        self.fc_sim_tranloc = nn.Linear(768, 768)
        self.fc_sim_tranglo = nn.Linear(768, 768)
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, image, text):
        """计算图像-文本的全局和局部对齐（输入顺序与GLAC相反）"""
        # 局部相似性表征
        image_aware_text, _ = self.CrossModalAlignment(image, text)

        sim_local = torch.pow(torch.sub(image, image_aware_text), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # 全局相似性表征
        image_cls_output = self.image_cls_pool(image)
        text_cls_output = self.text_cls_pool(text)
        sim_global = torch.pow(torch.sub(image_cls_output, text_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # 拼接全局和局部对齐
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)

        # 相似图推理
        sim_emb = self.SAF_module(sim_emb)

        return sim_emb

    def forward(self, image, text):
        path_prob = self.router(image)

        sim_emb = self.alignment(image, text)
        
        # 获取目标序列长度（文本长度）
        text_seq_len = text.size(1)
        
        # 将sim_emb扩展为与图像序列长度一致的tensor
        expanded_emb = sim_emb.unsqueeze(-2).expand(-1, image.size(1), -1)
        
        # 如果图像序列长度与文本序列长度不一致，进行调整
        if image.size(1) != text_seq_len:
            # 调整为与文本序列长度一致
            expanded_emb = F.interpolate(
                expanded_emb.transpose(1, 2),  # [B, D, L_img]
                size=text_seq_len,             # 目标长度L_text
                mode='linear'                  # 线性插值
            ).transpose(1, 2)                  # 变回[B, L_text, D]

        return expanded_emb, path_prob