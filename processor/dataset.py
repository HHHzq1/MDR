import random
import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
from collections import Counter


logger = logging.getLogger(__name__)



class MSDProcessor(object):
    def __init__(self, data_path, bert_name, clip_processor):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.clip_processor = clip_processor

    def load_from_file(self, mode="train"):

        logger.info("Loading data from {}".format(self.data_path[mode]))

        with open(self.data_path[mode], "r", encoding="utf-8") as f:
            dataset = json.load(f)  # 加载整个数据集

            raw_texts, raw_labels, imgs = [], [], []

            for index in range(0, len(dataset)):  # 一条一条地读取数据
                sample = dataset[index]

                img_id = sample['Pic_id']
                text = sample['Text']
                label = int(sample['Sentiment']) + 2 # 0-5
                assert 0 <= label <= 4, f"非法标签: {label}"
  
                # 将所有数据分别放到对应的列表中
                raw_texts.append(text)
                raw_labels.append(label)
                imgs.append(img_id)

        assert len(raw_texts) == len(raw_labels) == len(imgs), "{}, {}, {}".format(len(raw_texts), len(raw_labels), len(imgs))

        return {"texts": raw_texts, "labels": raw_labels, "imgs": imgs}


class MSDDataset(Dataset):
    """
    多模态情感检测数据集，支持三张图片输入
    - img_path: 原始广告图片路径
    - img_path2: Source区域截图路径
    - img_path3: Target区域截图路径
    """
    def __init__(self, processor, img_path, img_path2=None, img_path3=None, max_seq=128, mode="train"):
        self.processor = processor
        self.img_path = img_path      # 原始广告图片
        self.img_path2 = img_path2    # Source区域截图
        self.img_path3 = img_path3    # Target区域截图
        # 分词器
        self.tokenizer = self.processor.tokenizer
        self.data_dict = self.processor.load_from_file(mode)
        self.clip_processor = self.processor.clip_processor
        self.max_seq = max_seq
        # 统计Target图片的有效性
        self.path3_valid_count = 0
        self.path3_invalid_count = 0
        self.total_images = len(self.data_dict['imgs'])
        self.mode = mode


    def __len__(self):
        return len(self.data_dict['texts'])

    def __getitem__(self, idx):
        text, label = self.data_dict['texts'][idx], self.data_dict['labels'][idx]
        img_name = self.data_dict['imgs'][idx]

        tokens_text = self.tokenizer.tokenize(text)

        if len(tokens_text) > self.max_seq - 2:
            tokens_text = tokens_text[:self.max_seq - 2]

        tokens = ["[CLS]"] + tokens_text + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        # 处理三张图片：原图、Source截图、Target截图
        images = []
        
        # 图片1：原始广告图片（必须提供）
        img_path1 = os.path.join(self.img_path, img_name)
        try:
            image1 = Image.open(img_path1).convert('RGB')
            image1 = self.clip_processor(images=image1, return_tensors='pt')['pixel_values'].squeeze()
        except:
            fallback_path = os.path.join(self.img_path, 'inf.png')
            image1 = Image.open(fallback_path).convert('RGB')
            image1 = self.clip_processor(images=image1, return_tensors='pt')['pixel_values'].squeeze()
        images.append(image1)
        
        # 图片2：Source区域截图（可选，不存在时使用原图）
        if self.img_path2 is not None:
            img_path2 = os.path.join(self.img_path2, img_name)
            try:
                image2 = Image.open(img_path2).convert('RGB')
                image2 = self.clip_processor(images=image2, return_tensors='pt')['pixel_values'].squeeze()
            except:
                # 如果Source截图不存在，使用原图替代
                image2 = image1
            images.append(image2)
        else:
            # 如果未提供Source路径，使用原图
            images.append(image1)
        
        # 图片3：Target区域截图（可选，不存在时使用原图）
        if self.img_path3 is not None:
            img_path3 = os.path.join(self.img_path3, img_name)
            try:
                image3 = Image.open(img_path3).convert('RGB')
                image3 = self.clip_processor(images=image3, return_tensors='pt')['pixel_values'].squeeze()
                self.path3_valid_count += 1
                if idx == self.total_images - 1:
                    print(f"[{self.mode}] Target截图有效: {self.path3_valid_count}/{self.total_images}, 使用原图替代: {self.path3_invalid_count}")
            except:
                # 如果Target截图不存在，使用原图替代
                image3 = image1
                self.path3_invalid_count += 1
                if idx == self.total_images - 1:
                    print(f"[{self.mode}] Target截图有效: {self.path3_valid_count}/{self.total_images}, 使用原图替代: {self.path3_invalid_count}")
            images.append(image3)
        else:
            # 如果未提供Target路径，使用原图
            images.append(image1)
        
        img_mask = [1] * 50

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), \
               torch.tensor(img_mask), torch.tensor(label), images[0], images[1], images[2]

