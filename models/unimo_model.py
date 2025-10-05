"""
MDR_ads main model and loss function
"""

import torch
from torch import nn
from .modeling_unimo import UnimoModel
from torch.nn import CrossEntropyLoss


class UnimoModelF(nn.Module):
    """
    MDR_ads main model
    Supports three image inputs (original, Source crop, Target crop)
    Outputs 5-class sentiment prediction
    """
    def __init__(self, args, vision_config, text_config):
        super(UnimoModelF, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(args, vision_config, text_config)
        self.fc = nn.Linear(self.text_config.hidden_size, 5)  # 5-class classification

        self.CE_Loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images, images2=None, images3=None):
        """
        Args:
            input_ids: Text input IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Labels
            images: Original advertisement images
            images2: Source region crops (optional)
            images3: Target region crops (optional)
        """
        # If images2 and images3 are not provided, use original images
        if images2 is None:
            images2 = images
        if images3 is None:
            images3 = images
            
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values=images,
                            pixel_values2=images2,
                            pixel_values3=images3,
                            return_dict=True
                            )
        pool_out = output.pooler_output
        # 5-class head (bsz, 768) -> (bsz, 5)
        final_output = self.fc(pool_out)

        loss = self.CE_Loss(final_output, labels.long())

        return (loss, final_output)
