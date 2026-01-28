"""
Inference Script for MultiStain-GAN

Usage:
    python scripts/inference.py --input image.png --target ER --output output.png
    python scripts/inference.py --input image.png --all --output-dir outputs/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import numpy as np

from models import Generator, StyleEncoder, MappingNetwork
from data.transforms import get_val_transforms, AlbumentationsWrapper
from utils.visualization import tensor_to_pil, create_multi_domain_visualization


DOMAINS = ['ER', 'PR', 'Ki67', 'HER2']
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}


def load_models(checkpoint_path: str, device: torch.device):
    """Load trained models from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    
    generator = Generator(
        image_size=model_cfg.get('image_size', 256),
        style_dim=model_cfg.get('style_dim', 64),
        enc_channels=model_cfg.get('generator', {}).get('enc_channels', [64, 128, 256, 512]),
        num_res_blocks=model_cfg.get('generator', {}).get('num_res_blocks', 6),
        vit_blocks=model_cfg.get('generator', {}).get('vit_blocks', 4),
        vit_heads=model_cfg.get('generator', {}).get('vit_heads', 8),
    ).to(device)
    
    mapping_network = MappingNetwork(
        latent_dim=model_cfg.get('latent_dim', 16),
        style_dim=model_cfg.get('style_dim', 64),
        num_domains=model_cfg.get('num_domains', 4),
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator'])
    mapping_network.load_state_dict(checkpoint['mapping_network'])
    
    generator.eval()
    mapping_network.eval()
    
    return generator, mapping_network, model_cfg


def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess input image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    # Normalize to [-1, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


@torch.no_grad()
def generate(
    generator,
    mapping_network,
    source: torch.Tensor,
    target_domain: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate virtual stain for target domain."""
    source = source.to(device)
    batch_size = source.size(0)
    
    # Generate random style
    z = torch.randn(batch_size, 16, device=device)  # latent_dim = 16
    domain = torch.tensor([target_domain], device=device)
    style = mapping_network(z, domain)
    
    # Generate
    output = generator(source, style)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='MultiStain-GAN Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input H&E image path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pth',
                        help='Model checkpoint path')
    parser.add_argument('--target', type=str, choices=DOMAINS,
                        help='Target stain domain')
    parser.add_argument('--all', action='store_true',
                        help='Generate all stain types')
    parser.add_argument('--output', type=str,
                        help='Output image path')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for --all mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    if not args.target and not args.all:
        parser.error("Either --target or --all must be specified")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading models from {args.checkpoint}")
    generator, mapping_network, config = load_models(args.checkpoint, device)
    
    image_size = config.get('image_size', 256)
    
    # Load input image
    print(f"Processing: {args.input}")
    source = preprocess_image(args.input, image_size)
    
    if args.all:
        # Generate all domains
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_name = Path(args.input).stem
        generated = {}
        
        for domain in DOMAINS:
            print(f"Generating {domain}...")
            idx = DOMAIN_TO_IDX[domain]
            output = generate(generator, mapping_network, source, idx, device)
            
            # Save individual
            out_path = output_dir / f"{input_name}_{domain}.png"
            tensor_to_pil(output[0]).save(out_path)
            print(f"  Saved: {out_path}")
            
            generated[domain] = output[0]
        
        # Save comparison
        comparison_path = output_dir / f"{input_name}_comparison.png"
        create_multi_domain_visualization(source[0], generated, comparison_path)
        print(f"Comparison saved: {comparison_path}")
        
    else:
        # Generate single domain
        idx = DOMAIN_TO_IDX[args.target]
        print(f"Generating {args.target}...")
        output = generate(generator, mapping_network, source, idx, device)
        
        # Save
        output_path = args.output or f"output_{args.target}.png"
        tensor_to_pil(output[0]).save(output_path)
        print(f"Saved: {output_path}")
    
    print("Done!")


if __name__ == '__main__':
    main()
