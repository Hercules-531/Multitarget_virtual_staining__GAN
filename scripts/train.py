"""
Training Script for MultiStain-GAN

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume checkpoints/step_10000.pth
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from omegaconf import OmegaConf

from data.dataset import get_dataloader
from data.transforms import get_train_transforms, get_val_transforms, AlbumentationsWrapper
from trainers.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train MultiStain-GAN')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with reduced data')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Print config
    print("=" * 50)
    print("MultiStain-GAN Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Device: {config.get('device', 'cuda')}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print("=" * 50)
    
    # Data
    data_cfg = config.get('data', {})
    image_size = config.get('model', {}).get('image_size', 256)
    
    train_transform = AlbumentationsWrapper(get_train_transforms(image_size))
    val_transform = AlbumentationsWrapper(get_val_transforms(image_size))
    
    train_loader = get_dataloader(
        root_dir=data_cfg.get('root_dir', './data/ihc4bc'),
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        transform=train_transform,
        target_transform=train_transform,
        paired=True,
        paired_only=data_cfg.get('paired_only', False),
        paired_reference=data_cfg.get('paired_reference', False),
        image_size=image_size,
    )
    
    val_loader = get_dataloader(
        root_dir=data_cfg.get('root_dir', './data/ihc4bc'),
        split='val',
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        transform=val_transform,
        target_transform=val_transform,
        paired=True,
        paired_only=data_cfg.get('paired_only', False),
        paired_reference=data_cfg.get('paired_reference', False),
        image_size=image_size,
    )
    
    if args.debug:
        print("Running in debug mode with limited data...")
        # Limit data for debugging
        train_loader.dataset.samples = train_loader.dataset.samples[:100]
        val_loader.dataset.samples = val_loader.dataset.samples[:20]
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Trainer
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )
    
    # Train
    trainer.train(config['training']['epochs'])


if __name__ == '__main__':
    main()
