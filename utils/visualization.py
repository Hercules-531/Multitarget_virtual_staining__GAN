"""
Visualization Utilities

Image processing and visualization helpers.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    tensor = denormalize(tensor)
    tensor = tensor.clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    return Image.fromarray(tensor_to_numpy(tensor))


def make_grid_image(
    images: List[torch.Tensor],
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of image tensors
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to denormalize images
    
    Returns:
        PIL Image of the grid
    """
    from torchvision.utils import make_grid
    
    # Stack images
    grid = torch.stack(images, dim=0)
    
    # Make grid
    grid = make_grid(grid, nrow=nrow, padding=padding, normalize=normalize, value_range=(-1, 1))
    
    return tensor_to_pil(grid)


def save_comparison(
    source: torch.Tensor,
    generated: torch.Tensor,
    target: Optional[torch.Tensor],
    save_path: Union[str, Path],
    domain_name: str = "",
):
    """
    Save a side-by-side comparison of source, generated, and target.
    
    Args:
        source: Source H&E image
        generated: Generated IHC image
        target: Ground truth IHC image (optional)
        save_path: Path to save the comparison
        domain_name: Name of the target domain
    """
    fig, axes = plt.subplots(1, 3 if target is not None else 2, figsize=(12, 4))
    
    # Source
    axes[0].imshow(tensor_to_numpy(source))
    axes[0].set_title('H&E Input')
    axes[0].axis('off')
    
    # Generated
    axes[1].imshow(tensor_to_numpy(generated))
    axes[1].set_title(f'Generated {domain_name}')
    axes[1].axis('off')
    
    # Target
    if target is not None:
        axes[2].imshow(tensor_to_numpy(target))
        axes[2].set_title(f'Ground Truth {domain_name}')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_multi_domain_visualization(
    source: torch.Tensor,
    generated_images: dict,
    save_path: Union[str, Path],
):
    """
    Create visualization showing source and all generated domains.
    
    Args:
        source: Source H&E image
        generated_images: Dict mapping domain name to generated image
        save_path: Path to save visualization
    """
    num_domains = len(generated_images)
    fig, axes = plt.subplots(1, num_domains + 1, figsize=(4 * (num_domains + 1), 4))
    
    # Source
    axes[0].imshow(tensor_to_numpy(source))
    axes[0].set_title('H&E Input', fontsize=12)
    axes[0].axis('off')
    
    # Generated domains
    for i, (domain, img) in enumerate(generated_images.items()):
        axes[i + 1].imshow(tensor_to_numpy(img))
        axes[i + 1].set_title(f'{domain} (Generated)', fontsize=12)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class TensorBoardLogger:
    """TensorBoard logging utility."""
    
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        img = denormalize(image)
        self.writer.add_image(tag, img, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int, nrow: int = 4):
        """Log a grid of images."""
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        self.writer.add_image(tag, grid, step)
    
    def close(self):
        """Close the logger."""
        self.writer.close()
