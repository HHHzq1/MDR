# MDR_ads - Multimodal Dynamic Routing for Advertisement Sentiment Detection

Implementation of multimodal dynamic routing network for sentiment detection. This model adopts a single-branch architecture using 3 Cells for multimodal feature interaction.

## Model Architecture

### Key Features
- **Single-Branch Dynamic Routing**: Implements text-image feature interaction through 3 specialized Cells
- **Three-Cell Architecture**: 
  - GLAC (GlobalLocalAlignmentCell): Text domain processing
  - GESC (GlobalEnhancedSemanticCell): Image domain processing
  - R_GLAC (Reversed_GlobalLocalAlignmentCell): Complementary information processing
- **Multiple Image Input**: Supports three image inputs to adapt to complex advertising scenarios

### Base Models
- **Text Encoder**: BERT (bert-base-multilingual-uncased)
- **Image Encoder**: CLIP ViT (clip-vit-base-patch32)
- **Task**: 5-class sentiment detection

## Environment Requirements

Experimental Environment:
- Python 3.7.16
- PyTorch 1.7.1
- CUDA 11.2
- GPU: GeForce RTX 3090 (24GB)

Install Dependencies:
```bash
pip install -r requirements.txt
```

## Multiple Image Input Description

This model supports **three image inputs** for processing different visual elements in advertisements:

1. **Original Image (img_path)**: Complete original advertisement image
2. **Source Image (img_path2)**: Cropped source object region
3. **Target Image (img_path3)**: Cropped target object region

### Workflow
1. Read image filenames from JSON file
2. Search for corresponding images in three different folders
3. If an image doesn't exist in a folder, use the original image instead
4. Features from three images are **averaged and fused** before entering the interaction module

## Usage

### Basic Run
```bash
bash run.sh
```

### Specify Multiple Image Paths
```bash
python run.py --img_path data/images \
              --img_path2 data/source_images \
              --img_path3 data/target_images
```

### Main Parameters
Parameter Description:
- `img_path`: Original image folder path (**required**)
- `img_path2`: Source cropped image folder path (optional, uses original image if not provided)
- `img_path3`: Target cropped image folder path (optional, uses original image if not provided)

## Data Format

JSON data format example:
```json
{
    "Pic_id": "2223.jpg",
    "Text": "Advertisement text content",
    "Sentiment": 0
}
```

- `Pic_id`: Image filename (will be searched in three folders)
- `Text`: Advertisement text description
- `Sentiment`: Sentiment label (-2 to 2, automatically converted to 0-4 in code)

## Model Output

- 5-class sentiment prediction (label range: 0-4)
- Evaluation on validation and test sets during training

## Project Structure

```
MDR_ads/
├── models/              # Model definitions
│   ├── unimo_model.py   # Main model
│   ├── modeling_unimo.py # Encoder and interaction modules
│   ├── Cells.py         # Cell definitions
│   ├── DynamicInteraction.py  # Dynamic interaction layer
│   └── InteractionModule.py   # Interaction module
├── processor/           # Data processing
│   └── dataset.py       # Dataset loading
├── modules/             # Training modules
│   ├── train.py         # Trainer
│   └── metrics.py       # Evaluation metrics
├── run.py               # Main run script
├── run.sh               # Run script
└── requirements.txt     # Dependencies

```

## Technical Details

### Feature Fusion
CLIP features from three images are fused through simple averaging:
```python
vision_output = (vision_emb1 + vision_emb2 + vision_emb3) / 3
```

### Dynamic Routing
Adaptive feature interaction through learnable path probabilities:
- Each Cell outputs features and path probabilities
- Path probabilities are normalized for weighted fusion
- Supports multi-step iterative optimization

### Loss Function
- Cross-Entropy Loss (CrossEntropyLoss)

## License

This project is for research purposes only.