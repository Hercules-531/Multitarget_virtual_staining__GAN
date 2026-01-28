"""
Evaluation Metrics for Virtual Staining

Includes FID, SSIM, and LPIPS metrics.
"""

import torch
import numpy as np
from typing import Optional, List
from PIL import Image


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,  # For [-1, 1] normalized images
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image tensor (B, C, H, W) in range [-1, 1]
        img2: Second image tensor (B, C, H, W) in range [-1, 1]
        window_size: Size of Gaussian window
        data_range: Range of pixel values (2.0 for [-1, 1])
    
    Returns:
        Mean SSIM value
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
        ssim_metric = ssim_metric.to(img1.device)
        return ssim_metric(img1, img2).item()
    except ImportError:
        # Fallback implementation
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = img1.var()
        sigma2_sq = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.item()


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = 'alex',
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1: First image tensor (B, C, H, W) in range [-1, 1]
        img2: Second image tensor (B, C, H, W) in range [-1, 1]
        net: Network backbone ('alex' or 'vgg')
    
    Returns:
        Mean LPIPS value (lower is better)
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net).to(img1.device)
        with torch.no_grad():
            dist = loss_fn(img1, img2)
        return dist.mean().item()
    except ImportError:
        print("LPIPS not available. Install with: pip install lpips")
        return 0.0


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 50,
) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        real_images: Real images tensor (N, C, H, W) in range [-1, 1]
        fake_images: Generated images tensor (N, C, H, W) in range [-1, 1]
        batch_size: Batch size for processing
    
    Returns:
        FID score (lower is better)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        
        # Normalize to [0, 255] uint8 as required by FID
        real_uint8 = ((real_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fake_uint8 = ((fake_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        
        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        fid = fid.to(real_images.device)
        
        # Update with real images
        for i in range(0, len(real_uint8), batch_size):
            batch = real_uint8[i:i+batch_size]
            fid.update(batch, real=True)
        
        # Update with fake images
        for i in range(0, len(fake_uint8), batch_size):
            batch = fake_uint8[i:i+batch_size]
            fid.update(batch, real=False)
        
        return fid.compute().item()
    except ImportError:
        print("torchmetrics FID not available.")
        return 0.0


class EvaluationMetrics:
    """Container for all evaluation metrics."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.ssim_values = []
        self.lpips_values = []
        self.real_features = []
        self.fake_features = []
    
    def update(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ):
        """Update metrics with a batch of images."""
        # SSIM
        ssim_val = compute_ssim(fake, real)
        self.ssim_values.append(ssim_val)
        
        # LPIPS
        lpips_val = compute_lpips(fake, real)
        self.lpips_values.append(lpips_val)
    
    def compute(self) -> dict:
        """Compute final metrics."""
        results = {}
        
        if self.ssim_values:
            results['ssim'] = np.mean(self.ssim_values)
            results['ssim_std'] = np.std(self.ssim_values)
        
        if self.lpips_values:
            results['lpips'] = np.mean(self.lpips_values)
            results['lpips_std'] = np.std(self.lpips_values)
        
        return results


def evaluate_model(
    generator,
    style_encoder,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
) -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        generator: Trained generator model
        style_encoder: Trained style encoder
        dataloader: Test data loader
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate
    
    Returns:
        Dictionary of metric values
    """
    generator.eval()
    style_encoder.eval()
    
    metrics = EvaluationMetrics(device)
    
    all_real = []
    all_fake = []
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            
            source = batch['source'].to(device)
            reference = batch['reference'].to(device)
            target_domain = batch['target_domain'].to(device)
            target = batch.get('target')
            
            # Generate
            style = style_encoder(reference, target_domain)
            fake = generator(source, style)
            
            # Use target if available, otherwise use reference
            real = target.to(device) if target is not None else reference
            
            # Update metrics
            metrics.update(real, fake)
            
            # Collect for FID
            all_real.append(real.cpu())
            all_fake.append(fake.cpu())
            
            count += source.size(0)
    
    results = metrics.compute()
    
    # Compute FID
    if all_real and all_fake:
        all_real = torch.cat(all_real, dim=0)
        all_fake = torch.cat(all_fake, dim=0)
        results['fid'] = compute_fid(all_real, all_fake)
    
    generator.train()
    style_encoder.train()
    
    return results
