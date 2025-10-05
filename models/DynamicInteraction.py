"""
MDR_ads Dynamic Interaction Layer
Single-branch architecture using 3 Cells for multimodal feature interaction
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
    """Add two dimensions at the end"""
    return x.unsqueeze(-1).unsqueeze(-1)


def unsqueeze3d(x):
    """Add three dimensions at the end"""
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def clones(module, N):
    """Generate N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicInteraction_Layer0(nn.Module):
    """
    Dynamic Interaction Layer 0
    Uses 3 Cells for initial feature interaction:
    - GLAC: Text domain global-local alignment
    - GESC: Image domain global enhanced semantics
    - R_GLAC: Complementary information processing
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = 3  # Fixed at 3 Cells
        self.num_out_path = num_out_path
        
        # Three core Cells
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)  # Text domain
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)  # Image domain
        self.r_glac = Reversed_GlobalLocalAlignmentCell(args, num_out_path)  # Complementary

    def forward(self, text, image):
        """
        Args:
            text: Text features (bsz, seq_len, dim)
            image: Image features (bsz, num_patches, dim)
        Returns:
            aggr_res_lst: List of aggregated features
            all_path_prob: Path probabilities (bsz, num_out_path, num_cell)
        """
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        
        # Forward computation of three Cells
        emb_lst[0], path_prob[0] = self.glac(text, image)  # Text domain
        emb_lst[1], path_prob[1] = self.gesc(text, image)  # Image domain
        emb_lst[2], path_prob[2] = self.r_glac(image, text)  # Complementary

        # Gate mask: use skip connection when total path probability is too small
        gate_mask = (sum(path_prob) < self.threshold).float()
        
        # Normalize path probabilities
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        # Aggregate outputs from all paths
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
    Dynamic Interaction Intermediate Layer
    Continues feature interaction based on output from previous layer
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = 3  # Fixed at 3 Cells
        self.num_out_path = num_out_path
        
        # Three core Cells
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)
        self.r_glac = Reversed_GlobalLocalAlignmentCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):
        """
        Args:
            ref_wrd: Feature list from previous layer
            text: Original text features
            image: Original image features
        Returns:
            aggr_res_lst: List of aggregated features
            all_path_prob: Path probabilities
        """
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        
        # Forward computation of three Cells (using output from previous layer)
        emb_lst[0], path_prob[0] = self.glac(ref_wrd[0], image)
        emb_lst[1], path_prob[1] = self.gesc(ref_wrd[1], image)
        emb_lst[2], path_prob[2] = self.r_glac(ref_wrd[2], text)
        
        if self.num_out_path == 1:
            # Single output path mode
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
            # Multiple output paths mode
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