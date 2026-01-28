"""
Evaluation Script for MultiStain-GAN

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth
    python scripts/evaluate.py --checkpoint checkpoints/final.pth --num-samples 500
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from data.dataset import get_dataloader
from data.transforms import get_val_transforms, AlbumentationsWrapper
from models import Generator, StyleEncoder
from utils.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate MultiStain-GAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/ihc4bc',
                        help='Path to dataset')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    model_cfg = config.get('model', {})
    
    # Initialize models
    generator = Generator(
        image_size=model_cfg.get('image_size', 256),
        style_dim=model_cfg.get('style_dim', 64),
        enc_channels=model_cfg.get('generator', {}).get('enc_channels', [64, 128, 256, 512]),
        num_res_blocks=model_cfg.get('generator', {}).get('num_res_blocks', 6),
        vit_blocks=model_cfg.get('generator', {}).get('vit_blocks', 4),
        vit_heads=model_cfg.get('generator', {}).get('vit_heads', 8),
    ).to(device)
    
    style_encoder = StyleEncoder(
        image_size=model_cfg.get('image_size', 256),
        style_dim=model_cfg.get('style_dim', 64),
        num_domains=model_cfg.get('num_domains', 4),
    ).to(device)
    
    # Load weights
    generator.load_state_dict(checkpoint['generator'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    
    print("Models loaded successfully")
    
    # Data loader
    image_size = model_cfg.get('image_size', 256)
    transform = AlbumentationsWrapper(get_val_transforms(image_size))
    
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        num_workers=4,
        transform=transform,
        image_size=image_size,
    )
    
    print(f"Evaluating on {min(args.num_samples, len(test_loader.dataset))} samples...")
    
    # Evaluate
    results = evaluate_model(
        generator=generator,
        style_encoder=style_encoder,
        dataloader=test_loader,
        device=device,
        num_samples=args.num_samples,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric.upper()}: {value:.4f}")
    
    print("=" * 50)
    
    # Save results
    results_path = Path(args.checkpoint).parent / 'eval_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
