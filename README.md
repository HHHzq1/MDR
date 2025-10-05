# MDR_ads - Multimodal Dynamic Routing for Advertisement Sentiment Detection

多模态动态路由网络用于情感检测的实现。本模型采用单分支架构，使用3个Cell进行多模态特征交互。

## 模型架构

### 核心特点
- **单分支动态路由**: 通过3个专门的Cell实现文本-图像特征交互
- **三Cell架构**: 
  - GLAC (GlobalLocalAlignmentCell): 文本域处理
  - GESC (GlobalEnhancedSemanticCell): 图像域处理
  - R_GLAC (Reversed_GlobalLocalAlignmentCell): 互补信息处理
- **多图片输入**: 支持三张图片输入，适应复杂广告场景

### 基础模型
- **文本编码器**: BERT (bert-base-multilingual-uncased)
- **图像编码器**: CLIP ViT (clip-vit-base-patch32)
- **任务**: 5分类情感检测

## 环境要求

实验环境：
- Python 3.7.16
- PyTorch 1.7.1
- CUDA 11.2
- GPU: GeForce RTX 3090 (24GB)

安装依赖：
```bash
pip install -r requirements.txt
```

## 多图片输入说明

本模型支持**三张图片输入**，用于处理广告中的不同视觉元素：

1. **原图片 (img_path)**: 完整的广告原图
2. **Source图片 (img_path2)**: 截取的源对象区域
3. **Target图片 (img_path3)**: 截取的目标对象区域

### 工作流程
1. 从JSON文件中读取图片文件名
2. 在三个不同文件夹中查找对应的图片
3. 若某文件夹中不存在该图片，则使用原图片代替
4. 三张图片的特征在进入交互模块前会被**平均融合**

## 使用方法

### 基础运行
```bash
bash run.sh
```

### 指定多图片路径
```bash
python run.py --img_path /path/to/original/images \
              --img_path2 /path/to/source/images \
              --img_path3 /path/to/target/images
```

### 主要参数
参数说明：
- `img_path`: 原图片文件夹路径（**必需**）
- `img_path2`: Source截图文件夹路径（可选，不提供时使用原图）
- `img_path3`: Target截图文件夹路径（可选，不提供时使用原图）

## 数据格式

JSON数据格式示例：
```json
{
    "Pic_id": "2223.jpg",
    "Text": "广告文本内容",
    "Sentiment": 0
}
```

- `Pic_id`: 图片文件名（会在三个文件夹中查找）
- `Text`: 广告文本描述
- `Sentiment`: 情感标签（-2到2，代码中会自动转换为0-4）

## 模型输出

- 5分类情感预测（标签范围：0-4）
- 训练过程中会在验证集和测试集上评估

## 项目结构

```
MDR_ads/
├── models/              # 模型定义
│   ├── unimo_model.py   # 主模型
│   ├── modeling_unimo.py # 编码器和交互模块
│   ├── Cells.py         # Cell定义
│   ├── DynamicInteraction.py  # 动态交互层
│   └── InteractionModule.py   # 交互模块
├── processor/           # 数据处理
│   └── dataset.py       # 数据集加载
├── modules/             # 训练模块
│   ├── train.py         # 训练器
│   └── metrics.py       # 评估指标
├── run.py               # 主运行脚本
├── run.sh               # 运行脚本
└── requirements.txt     # 依赖包

```

## 技术细节

### 特征融合
三张图片的CLIP特征通过简单平均进行融合：
```python
vision_output = (vision_emb1 + vision_emb2 + vision_emb3) / 3
```

### 动态路由
通过可学习的路径概率实现自适应的特征交互：
- 每个Cell输出特征和路径概率
- 路径概率经过归一化后进行加权融合
- 支持多步迭代优化

### 损失函数
- 交叉熵损失（CrossEntropyLoss）

## License

本项目仅用于研究目的。

## 联系方式

如有问题，请提交Issue。