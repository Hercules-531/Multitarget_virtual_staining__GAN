"""
Gradio Web UI for MultiStain-GAN

Interactive interface for virtual staining inference.

Usage:
    python app/gradio_app.py
    python app/gradio_app.py --checkpoint path/to/model.pth --share
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict

# Import models
from models import Generator, MappingNetwork


DOMAINS = ['ER', 'PR', 'Ki67', 'HER2']
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}

# Global model references
generator = None
mapping_network = None
device = None
image_size = 256


def load_models(checkpoint_path: str = "checkpoints/final.pth"):
    """Load trained models."""
    global generator, mapping_network, device, image_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("The app will run in demo mode with random weights.")
        
        # Initialize with default config
        generator = Generator(
            image_size=256,
            style_dim=64,
        ).to(device)
        
        mapping_network = MappingNetwork(
            latent_dim=16,
            style_dim=64,
            num_domains=4,
        ).to(device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        model_cfg = config.get('model', {})
        image_size = model_cfg.get('image_size', 256)
        
        generator = Generator(
            image_size=image_size,
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
            num_layers=model_cfg.get('mapping_network', {}).get('num_layers', 4),
        ).to(device)
        
        generator.load_state_dict(checkpoint['generator'])
        mapping_network.load_state_dict(checkpoint['mapping_network'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    generator.eval()
    mapping_network.eval()


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Preprocess input image."""
    if image is None:
        return None
    
    # Resize
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    # Normalize to [-1, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Postprocess output tensor to numpy image."""
    img = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    img = img.clamp(0, 1)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


@torch.no_grad()
def generate_stain(
    input_image: np.ndarray,
    target_domain: str,
) -> Tuple[np.ndarray, str]:
    """Generate virtual stain for a single domain."""
    if input_image is None:
        return None, "Please upload an image first."
    
    if generator is None:
        return None, "Model not loaded. Please check the checkpoint path."
    
    # Preprocess
    source = preprocess(input_image).to(device)
    
    # Get domain index
    domain_idx = DOMAIN_TO_IDX.get(target_domain, 0)
    
    # Generate random style
    z = torch.randn(1, 16, device=device)
    domain = torch.tensor([domain_idx], device=device)
    style = mapping_network(z, domain)
    
    # Generate
    output = generator(source, style)
    
    # Postprocess
    result = postprocess(output)
    
    return result, f"Generated {target_domain} stain successfully!"


@torch.no_grad()
def generate_all_stains(
    input_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate all virtual stains."""
    if input_image is None:
        return None, None, None, None, "Please upload an image first."
    
    if generator is None:
        return None, None, None, None, "Model not loaded."
    
    # Preprocess
    source = preprocess(input_image).to(device)
    
    results = []
    for domain in DOMAINS:
        domain_idx = DOMAIN_TO_IDX[domain]
        
        z = torch.randn(1, 16, device=device)
        domain_tensor = torch.tensor([domain_idx], device=device)
        style = mapping_network(z, domain_tensor)
        
        output = generator(source, style)
        results.append(postprocess(output))
    
    return results[0], results[1], results[2], results[3], "Generated all stains successfully!"


def create_ui():
    """Create Gradio interface."""
    
    with gr.Blocks(
        title="MultiStain-GAN Virtual Staining",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown("""
        # 🔬 MultiStain-GAN: Multi-Target Virtual Staining
        
        Transform H&E stained breast cancer tissue images into virtual IHC stains 
        (ER, PR, Ki67, HER2) using a single deep learning model.
        
        **How to use:**
        1. Upload an H&E stained tissue image
        2. Select a target stain or generate all stains
        3. Click Generate!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input H&E Image",
                    type="numpy",
                    height=300,
                )
                
                with gr.Row():
                    target_domain = gr.Dropdown(
                        choices=DOMAINS,
                        value="ER",
                        label="Target Stain",
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                
                generate_all_btn = gr.Button("Generate All Stains", variant="secondary")
                
                status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Stain",
                    type="numpy",
                    height=300,
                )
        
        gr.Markdown("### All Stains (Generate All)")
        
        with gr.Row():
            er_output = gr.Image(label="ER", type="numpy", height=200)
            pr_output = gr.Image(label="PR", type="numpy", height=200)
            ki67_output = gr.Image(label="Ki67", type="numpy", height=200)
            her2_output = gr.Image(label="HER2", type="numpy", height=200)
        
        # Event handlers
        generate_btn.click(
            fn=generate_stain,
            inputs=[input_image, target_domain],
            outputs=[output_image, status],
        )
        
        generate_all_btn.click(
            fn=generate_all_stains,
            inputs=[input_image],
            outputs=[er_output, pr_output, ki67_output, her2_output, status],
        )
        
        gr.Markdown("""
        ---
        ### About
        
        This model performs multi-domain image-to-image translation to generate 
        virtual immunohistochemistry (IHC) stains from H&E (Hematoxylin & Eosin) 
        stained tissue images.
        
        **Biomarkers:**
        - **ER** (Estrogen Receptor)
        - **PR** (Progesterone Receptor)  
        - **Ki67** (Proliferation marker)
        - **HER2** (Human Epidermal Growth Factor Receptor 2)
        
        **Architecture:** MultiStain-GAN (StarGAN v2 + ViT + Contrastive Learning)
        """)
    
    return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MultiStain-GAN Gradio App')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run on')
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    load_models(args.checkpoint)
    
    # Create and launch UI
    demo = create_ui()
    
    print(f"\nStarting Gradio app on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == '__main__':
    main()
