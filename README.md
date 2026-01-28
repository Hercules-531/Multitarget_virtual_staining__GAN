# Multi-Target Virtual Staining GAN

PyTorch implementation of MultiStain-GAN for translating H&E images to multiple IHC stains.

## Features

- 🎯 **Multi-domain translation**: Single model for ER, PR, Ki67, HER2
- 🔬 **Hybrid architecture**: StarGAN v2 + ViT blocks + Contrastive learning
- 🖼️ **High structural fidelity**: SSIM-based losses preserve tissue structure
- 🌐 **Gradio Web UI**: Easy inference interface

## Installation

```bash
# Clone and setup
cd Multitarget_virtual_staining__GAN
pip install -r requirements.txt

# Download dataset (requires Kaggle API key)
python data/download.py
```

## Usage

### Training

```bash
python scripts/train.py --config configs/config.yaml
```

### Inference

```bash
python scripts/inference.py --input path/to/he_image.png --target ER
```

### Web UI

```bash
python app/gradio_app.py
```

## Architecture

```
H&E Image → CNN Encoder → ViT Blocks → AdaIN (+ Style) → CNN Decoder → IHC Image
                                            ↑
                            Style Encoder / Mapping Network
```

## Dataset

[IHC4BC](https://www.kaggle.com/datasets/akbarnejad1991/ihc4bc-compressed) - 90K paired H&E/IHC breast cancer images

## Citation

If you use this code, please cite the original IHC4BC dataset paper.
