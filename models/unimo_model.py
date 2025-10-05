"""
MDR_ads 主模型和损失函数
"""

import torch
from torch import nn
from .modeling_unimo import UnimoModel
from torch.nn import CrossEntropyLoss


class UnimoModelF(nn.Module):
    """
    MDR_ads 主模型
    支持三张图片输入（原图、Source截图、Target截图）
    输出5分类情感预测
    """
    def __init__(self, args, vision_config, text_config):
        super(UnimoModelF, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(args, vision_config, text_config)
        self.fc = nn.Linear(self.text_config.hidden_size, 5)  # 5分类

        self.CE_Loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images, images2=None, images3=None):
        """
        Args:
            input_ids: 文本输入ID
            attention_mask: 注意力掩码
            token_type_ids: Token类型ID
            labels: 标签
            images: 原始广告图片
            images2: Source区域截图（可选）
            images3: Target区域截图（可选）
        """
        # 如果没有提供images2和images3，使用原图替代
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
        # 5分类头 (bsz, 768) -> (bsz, 5)
        final_output = self.fc(pool_out)

        loss = self.CE_Loss(final_output, labels.long())

        return (loss, final_output)
