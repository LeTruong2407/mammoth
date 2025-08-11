# KARMA: Kolmogorov-Arnold Radial Memory Architecture

<p align="center">
  <img width="800" height="200" src="docs/_static/mammoth_banner.svg" alt="Mammoth Banner">
</p>

<p align="center">
  <strong>KARMA: A Continual Learning Framework combining Kolmogorov-Arnold Classifier (KAC) with Incremental Task Arithmetic (ITA) and Second-Order Regularization</strong>
</p>

## 📖 Overview

KARMA (Kolmogorov-Arnold Radial Memory Architecture) is a novel continual learning framework that integrates:

- **Kolmogorov-Arnold Classifier (KAC)**: A classifier head based on Radial Basis Functions (RBFs)
- **Incremental Task Arithmetic (ITA)**: Method for composing models by averaging task-specific parameter changes
- **Second-Order Regularization**: Based on Fisher Information Matrix (FIM) to prevent catastrophic forgetting
- **Vision Transformer (ViT) Backbone**: Adapted using Low-Rank Adaptation (LoRA)

This implementation is built on top of the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd mammoth

# Create conda environment
conda create -n karma python=3.8 -y
conda activate karma

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for advanced features
pip install -r requirements-optional.txt
```

### 2. Basic Usage

Run KARMA on CUB-200 dataset with default settings:

```bash
python main.py --model second_order --dataset seq_cub200 --alpha_ita 0.02 --req_weight_cls 0.1 --simple_reg_weight_cls 0.001 --lr 0.0003 --batch_size 64
```

### 3. Advanced Configuration

```bash
# Run with LoRA fine-tuning (recommended)
python main.py --model second_order --dataset seq_cub200 \
    --alpha_ita 0.02 \
    --req_weight_cls 0.1 \
    --simple_reg_weight_cls 0.001 \
    --lr 0.0003 \
    --batch_size 64 \
    --n_epochs 50

# Run with full fine-tuning
python main.py --model second_order --dataset seq_cub200 \
    --tuning_style full \
    --alpha_ita 0.02 \
    --req_weight_cls 0.1 \
    --simple_reg_weight_cls 0.001 \
    --lr 0.0001 \
    --batch_size 32
```

## 📊 Supported Datasets

KARMA supports all datasets available in Mammoth:

- **CUB-200**: `seq_cub200` (10 tasks × 20 classes)
- **CIFAR-100**: `seq_cifar100` (10 tasks × 10 classes)  
- **ImageNet-R**: `seq_imagenet_r` (10 tasks × 20 classes)
- **MIT-67**: `seq_mit67` (10 tasks)
- **Caltech-256**: `seq_caltech256` (10 tasks)
- **RESISC45**: `seq_resisc45` (9 tasks × 5 classes)
- **CropDisease**: `seq_cropdisease` (7 tasks × 5 classes)

## ⚙️ Key Parameters

### Model Parameters
- `--model second_order`: Use the KARMA implementation
- `--tuning_style`: Fine-tuning strategy (`lora`, `full`, `ia3`)
- `--lora_r`: LoRA rank (default: 16)

### Regularization Parameters
- `--alpha_ita`: ITA regularization strength (default: 0.02)
- `--req_weight_cls`: Classifier regularization weight (default: 0.1)
- `--simple_reg_weight_cls`: Additional classifier regularization (default: 0.001)

### Training Parameters
- `--lr`: Learning rate (default: 0.0003)
- `--batch_size`: Batch size (default: 64)
- `--n_epochs`: Number of epochs per task (default: 50)



## 🧪 Experimental Features

## 📁 Project Structure

```
mammoth/
├── models/
│   ├── second_order.py          # Main KARMA implementation
│   ├── kac.py                   # KAC      # Advanced RBF implementations
│   └── lora_prototype_utils/    # LoRA integration
├── datasets/                    # Dataset implementations
├── backbone/                    # ViT backbone
├── utils/                       # Utility functions
├── main.py                      # Main 
```

## 🔬 Reproducing Results



### Custom Experiments
```bash
# Ablation study: without regularization
python main.py --model second_order --dataset seq_cub200 --alpha_ita 0.0 --req_weight_cls 0.0

# Different LoRA ranks
python main.py --model second_order --dataset seq_cub200 --lora_r 8
python main.py --model second_order --dataset seq_cub200 --lora_r 32
```

## 🐛 Troubleshooting





3. **Dataset Download Issues**
   ```bash
   # Manual dataset download
   python download_cub200.py
   ```

### Debug Mode
```bash
# Enable debug mode for detailed logging
python main.py --model second_order --dataset seq_cub200 --debug_mode 1
```



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



## 🔄 Updates

- **v1.0**: Initial KARMA implementation with KAC and ITA
- **v1.1**: Added RBF alternatives and advanced logging
- **v1.2**: Improved regularization and stability

---

**Note**: This implementation is based on the Mammoth framework. For more information about Mammoth, see the [original repository](https://github.com/aimagelab/mammoth). 
