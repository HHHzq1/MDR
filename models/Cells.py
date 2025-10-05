"""
MDR_ads Cell Module Definitions
Contains three core Cells:
- GlobalLocalAlignmentCell (GLAC): Text domain processing
- GlobalEnhancedSemanticCell (GESC): Image domain processing
- Reversed_GlobalLocalAlignmentCell (R_GLAC): Complementary information processing
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
    """Pooler to extract [CLS] token from BERT hidden states"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Extract the hidden state of the first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GlobalLocalAlignmentCell(nn.Module):
    """
    Global-Local Alignment Cell (GLAC) - Text Domain Processing
    Enhances feature representation by computing global and local similarity between text and image
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
        """Compute global and local alignment between text and image"""
        # Local similarity representation
        text_aware_image, _ = self.CrossModalAlignment(text, image)  # (bsz, seq_len, 768)

        sim_local = torch.pow(torch.sub(text, text_aware_image), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # Global similarity representation
        text_cls_output = self.text_cls_pool(text)
        image_cls_output = self.image_cls_pool(image)
        sim_global = torch.pow(torch.sub(text_cls_output, image_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # Concatenate global and local alignment
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)  # (bsz, seq_len+1, 768)

        # Similarity graph reasoning
        sim_emb = self.SAF_module(sim_emb)  # (bsz, 768)

        return sim_emb

    def forward(self, text, image):
        path_prob = self.router(text)
        
        sim_emb = self.alignment(text, image)
        sim_emb = sim_emb.unsqueeze(-2).expand(-1, text.size(1), -1)

        return sim_emb, path_prob


class GlobalEnhancedSemanticCell(nn.Module):
    """
    Global Enhanced Semantic Cell (GESC) - Image Domain Processing
    Fuses global semantic information of text and image through gating mechanism
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
        """Global gated fusion"""
        text_cls = self.text_cls_pool(text)  # (bsz, 768)
        image_cls = self.image_cls_pool(image)  # (bsz, 768)

        # Gating mechanism: global information alignment and fusion
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
    Reversed Global-Local Alignment Cell (R_GLAC) - Complementary Information Processing
    Opposite to GLAC, uses image as query and text as key-value for alignment
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
        """Compute global and local alignment between image and text (input order reversed from GLAC)"""
        # Local similarity representation
        image_aware_text, _ = self.CrossModalAlignment(image, text)

        sim_local = torch.pow(torch.sub(image, image_aware_text), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # Global similarity representation
        image_cls_output = self.image_cls_pool(image)
        text_cls_output = self.text_cls_pool(text)
        sim_global = torch.pow(torch.sub(image_cls_output, text_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # Concatenate global and local alignment
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)

        # Similarity graph reasoning
        sim_emb = self.SAF_module(sim_emb)

        return sim_emb

    def forward(self, image, text):
        path_prob = self.router(image)

        sim_emb = self.alignment(image, text)
        
        # Get target sequence length (text length)
        text_seq_len = text.size(1)
        
        # Expand sim_emb to match image sequence length
        expanded_emb = sim_emb.unsqueeze(-2).expand(-1, image.size(1), -1)
        
        # If image sequence length doesn't match text sequence length, adjust
        if image.size(1) != text_seq_len:
            # Adjust to match text sequence length
            expanded_emb = F.interpolate(
                expanded_emb.transpose(1, 2),  # [B, D, L_img]
                size=text_seq_len,             # Target length L_text
                mode='linear'                  # Linear interpolation
            ).transpose(1, 2)                  # Back to [B, L_text, D]

        return expanded_emb, path_prob