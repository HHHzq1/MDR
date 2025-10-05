"""
MDR_ads Interaction Module
Single-branch dynamic routing module using 3 Cells for multimodal feature interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer


class MultimodalInteractionModule(nn.Module):
    """
    Multimodal Interaction Module
    Achieves deep text-image feature interaction through multi-layer dynamic routing
    - Layer0: Initial interaction layer
    - Layer1-N: Intermediate interaction layers (multiple layers possible)
    - Layer_final: Final aggregation layer
    """
    def __init__(self, args, num_layer_routing=3, num_cells=3, path_hid=128):
        """
        Args:
            args: Parameter configuration
            num_layer_routing: Number of routing layers (default 3)
            num_cells: Number of Cells (fixed at 3)
            path_hid: Path hidden layer dimension
        """
        super(MultimodalInteractionModule, self).__init__()
        self.args = args
        self.num_cells = 3  # Fixed at 3 Cells: GLAC, GESC, R_GLAC
        
        # Layer 0: Initial interaction (3 Cells -> 3 output paths)
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(args, self.num_cells, self.num_cells)
        
        # Intermediate layers: Continue interaction (3 Cells -> 3 output paths)
        self.dynamic_itr_l1 = nn.ModuleList([
            DynamicInteraction_Layer(args, self.num_cells, self.num_cells) 
            for i in range(num_layer_routing - 2)
        ])
        
        # Final layer: Aggregate to single output (3 Cells -> 1 output path)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(args, self.num_cells, 1)
        
        # Path mapping layer (for similarity computation)
        total_paths = self.num_cells ** 2 * (num_layer_routing - 1) + self.num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)

    def forward(self, text, image):
        """
        Args:
            text: Text features (bsz, seq_len, dim)
            image: Image features (bsz, num_patches, dim)
        Returns:
            pairs_emb_lst: List of interaction features
            sim_paths: Path similarity matrix (bsz, bsz)
        """
        mid_paths = []
        
        # Layer 0: Initial interaction
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(text, image)
        
        # Intermediate layers: Layer-by-layer interaction
        for module in self.dynamic_itr_l1:
            pairs_emb_lst, paths_l1 = module(pairs_emb_lst, text, image)
            mid_paths.append(paths_l1)
            
        # Final layer: Aggregation
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, text, image)
        
        n_img, n_stc = paths_l2.size()[:2]
        
        # Organize path probabilities
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
        
        # Compute path similarity matrix
        sim_paths = torch.matmul(paths.squeeze(-2), paths.squeeze(-2).transpose(-1, -2))
        
        return pairs_emb_lst, sim_paths