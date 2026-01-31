"""
Loss Functions for MultiStain-GAN

Comprehensive losses for multi-domain virtual staining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import torchvision.models as models


class AdversarialLoss(nn.Module):
    """Hinge adversarial loss with label smoothing for stability."""
    
    def __init__(self, loss_type: str = "hinge", label_smoothing: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        real_scores: List[torch.Tensor],
        fake_scores: List[torch.Tensor],
        mode: str = "discriminator",
    ) -> torch.Tensor:
        """
        Compute adversarial loss with label smoothing.
        
        Args:
            real_scores: List of real image scores from multi-scale D
            fake_scores: List of fake image scores from multi-scale D
            mode: 'discriminator' or 'generator'
        
        Returns:
            Loss value
        """
        loss = 0.0
        
        if self.loss_type == "hinge":
            # Soft hinge loss with label smoothing for discriminator stability
            # This prevents D from being overconfident
            if mode == "discriminator":
                # Label smoothing: soften the margin (1.0 -> 0.9)
                margin_real = 1.0 - self.label_smoothing  # 0.9 instead of 1.0
                margin_fake = -1.0 + self.label_smoothing  # -0.9 instead of -1.0
                for real, fake in zip(real_scores, fake_scores):
                    # Clamp scores to prevent NaN
                    real = torch.clamp(real, -100.0, 100.0)
                    fake = torch.clamp(fake, -100.0, 100.0)
                    # Softened margins for label smoothing
                    loss += torch.mean(F.relu(margin_real - real))
                    loss += torch.mean(F.relu(fake - margin_fake))
            else:  # generator
                for fake in fake_scores:
                    # Clamp scores to prevent NaN from extreme values
                    fake = torch.clamp(fake, -100.0, 100.0)
                    # Generator wants fake scores to be high
                    loss += -torch.mean(fake)
        else:  # LSGAN (fallback)
            if mode == "discriminator":
                # Label smoothing: use 0.9 instead of 1.0 for real
                real_label = 1.0 - self.label_smoothing
                for real, fake in zip(real_scores, fake_scores):
                    loss += torch.mean((real - real_label) ** 2)
                    loss += torch.mean(fake ** 2)
            else:  # generator
                for fake in fake_scores:
                    loss += torch.mean((fake - 1) ** 2)
        
        return loss / max(len(fake_scores), 1)


class CycleLoss(nn.Module):
    """Cycle consistency loss."""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cycle consistency loss."""
        return self.l1(reconstructed, original)


class StyleReconstructionLoss(nn.Module):
    """Style reconstruction loss."""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        style_target: torch.Tensor,
        style_generated: torch.Tensor,
    ) -> torch.Tensor:
        """Compute style reconstruction loss."""
        return self.l1(style_generated, style_target)


class ColorHistogramLoss(nn.Module):
    """
    Color histogram matching loss.
    
    Forces generated images to have similar color distribution as target domain.
    This is CRITICAL for preventing color collapse in stain translation.
    """
    
    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
    
    def compute_histogram(self, img: torch.Tensor) -> torch.Tensor:
        """Compute soft color histogram for each channel."""
        # img: (B, 3, H, W) in range [-1, 1]
        img = (img + 1) / 2  # to [0, 1]
        img = img.clamp(0, 1)
        
        B, C, H, W = img.shape
        img_flat = img.view(B, C, -1)  # (B, 3, H*W)
        
        # Create soft histogram using Gaussian kernels
        bin_centers = torch.linspace(0, 1, self.num_bins, device=img.device)
        bin_width = 1.0 / self.num_bins
        sigma = bin_width * 1.5  # Soft assignment
        
        # (B, 3, H*W, 1) - (num_bins,) -> (B, 3, H*W, num_bins)
        diff = img_flat.unsqueeze(-1) - bin_centers.view(1, 1, 1, -1)
        weights = torch.exp(-0.5 * (diff / sigma) ** 2)
        
        # Normalize weights and sum to get histogram
        hist = weights.sum(dim=2)  # (B, 3, num_bins)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
        
        return hist
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute color histogram matching loss."""
        hist_gen = self.compute_histogram(generated)
        hist_tgt = self.compute_histogram(target)
        
        # L1 distance between histograms (per channel, then average)
        loss = F.l1_loss(hist_gen, hist_tgt)
        return loss


class ColorMomentsLoss(nn.Module):
    """
    Color moments (mean, std, skewness) matching loss.
    
    Simpler than histogram but effective for matching color statistics.
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_moments(self, img: torch.Tensor) -> torch.Tensor:
        """Compute mean, std, skewness for each channel."""
        # img: (B, 3, H, W)
        B, C, H, W = img.shape
        img_flat = img.view(B, C, -1)  # (B, 3, H*W)
        
        mean = img_flat.mean(dim=-1)  # (B, 3)
        std = img_flat.std(dim=-1) + 1e-6  # (B, 3)
        
        # Skewness
        centered = img_flat - mean.unsqueeze(-1)
        skew = (centered ** 3).mean(dim=-1) / (std ** 3 + 1e-6)
        
        # Concatenate moments
        moments = torch.cat([mean, std, skew], dim=-1)  # (B, 9)
        return moments
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute color moments matching loss."""
        mom_gen = self.compute_moments(generated)
        mom_tgt = self.compute_moments(target)
        
        return F.l1_loss(mom_gen, mom_tgt)


class StyleDiversificationLoss(nn.Module):
    """Style diversification loss to encourage diverse outputs, especially colors."""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        style1: torch.Tensor,
        style2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage diverse outputs for different style codes.
        Includes both pixel-level and color distribution diversity.
        """
        # Pixel-level difference
        pixel_diff = self.l1(image1, image2)
        
        # Color mean difference (encourage different colors for different styles)
        mean1 = image1.mean(dim=[2, 3])  # (B, 3)
        mean2 = image2.mean(dim=[2, 3])
        color_diff = (mean1 - mean2).abs().mean()
        
        # Combined diversity loss (negative because we want to maximize)
        total_diff = pixel_diff + 2.0 * color_diff  # Emphasize color diversity
        return -total_diff


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Computes L1 distance between VGG feature maps.
    """
    
    def __init__(self, layers: List[int] = [3, 8, 17, 26]):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        self.l1 = nn.L1Loss()
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        x = x.clamp(-1.0, 1.0)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss."""
        # Disable autocast and run VGG in float32 (mixed precision fix)
        input_dtype = generated.dtype
        device = generated.device
        
        # Ensure VGG blocks are on the same device
        if next(self.blocks[0].parameters()).device != device:
            self.blocks = self.blocks.to(device)
        
        with torch.amp.autocast('cuda', enabled=False):
            generated = self._normalize(generated.float())
            target = self._normalize(target.float())
            
            loss = 0.0
            x_gen = generated
            x_tar = target
            
            for block in self.blocks:
                x_gen = block(x_gen)
                x_tar = block(x_tar)
                loss += self.l1(x_gen, x_tar)
            
            loss = loss / len(self.blocks)
        
        return loss.to(input_dtype)


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss.
    
    Encourages structural preservation in generated images.
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3, eps: float = 1e-6):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.eps = eps
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        # Ensure window is on the same device and dtype
        window = self.window.to(generated.device, generated.dtype)
        
        # Normalize to [0, 1]
        generated = generated.clamp(-1.0, 1.0)
        target = target.clamp(-1.0, 1.0)
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(generated, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(generated ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(generated * target, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        sigma1_sq = sigma1_sq.clamp(min=0.0)
        sigma2_sq = sigma2_sq.clamp(min=0.0)
        
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        denom = denom + self.eps
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denom
        
        return 1 - ssim_map.mean()


class PatchNCELoss(nn.Module):
    """
    Patch-wise Contrastive Loss (NCE).
    
    Encourages structural consistency by comparing encoder features from source
    with encoder features from the generated fake image.
    """
    
    def __init__(
        self,
        nce_layers: List[int] = [1, 2, 3],  # Use mid-level encoder features
        num_patches: int = 256,
        temperature: float = 0.1,  # Increased from 0.07 for stability
        logit_clip: float = 30.0,  # Reduced from 50.0 for stability
    ):
        super().__init__()
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.temperature = max(temperature, 0.05)  # Floor at 0.05 to prevent NaN
        self.logit_clip = logit_clip
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
        # MLP heads for each layer - will be initialized on first forward
        self.mlps = nn.ModuleDict()
    
    def _get_or_create_mlp(self, layer_idx: int, nc: int, device: torch.device) -> nn.Module:
        """Get or create MLP head for a specific layer."""
        key = str(layer_idx)
        if key not in self.mlps:
            self.mlps[key] = nn.Sequential(
                nn.Linear(nc, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            ).to(device)
        return self.mlps[key]
    
    def forward(
        self,
        source_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss between source encoder features and fake encoder features.
        
        Args:
            source_features: Multi-scale encoder features from source image
            fake_features: Multi-scale encoder features from generated fake image
        
        Returns:
            NCE loss value - lower means better structural consistency
        """
        # Select which feature layers to use
        layers_to_use = [i for i in self.nce_layers if i < len(source_features)]
        if not layers_to_use:
            layers_to_use = list(range(min(3, len(source_features))))
        
        total_loss = 0.0
        num_layers = 0
        
        for layer_idx in layers_to_use:
            feat_s = source_features[layer_idx]
            feat_f = fake_features[layer_idx]
            
            B, C, H, W = feat_s.shape
            device = feat_s.device
            
            # Sample random patches - create indices on same device
            num_patches = min(self.num_patches, H * W)
            indices = torch.randperm(H * W, device=device)[:num_patches]
            
            # Reshape to (B, C, H*W)
            feat_s_flat = feat_s.flatten(2)
            feat_f_flat = feat_f.flatten(2)
            
            # Sample patches using indices
            feat_s_patches = feat_s_flat[:, :, indices]  # (B, C, num_patches)
            feat_f_patches = feat_f_flat[:, :, indices]
            
            # Project through MLP: (B, C, num_patches) -> (B*num_patches, 256)
            feat_s_proj = feat_s_patches.permute(0, 2, 1).reshape(-1, C)
            feat_f_proj = feat_f_patches.permute(0, 2, 1).reshape(-1, C)
            
            mlp = self._get_or_create_mlp(layer_idx, C, device)
            feat_s_proj = mlp(feat_s_proj)
            feat_f_proj = mlp(feat_f_proj)
            
            # Normalize
            feat_s_proj = F.normalize(feat_s_proj, dim=1)
            feat_f_proj = F.normalize(feat_f_proj, dim=1)
            
            # Compute similarities: each fake patch should match corresponding source patch
            logits = torch.mm(feat_f_proj, feat_s_proj.t()) / self.temperature
            # Subtract max for numerical stability
            logits = logits - logits.max(dim=1, keepdim=True).values
            if self.logit_clip is not None:
                logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
            
            # Labels: positive pairs are on diagonal (patch i in fake should match patch i in source)
            labels = torch.arange(logits.size(0), device=device)
            
            loss = self.cross_entropy(logits, labels)
            total_loss = total_loss + loss.mean()
            num_layers += 1
        
        return total_loss / max(num_layers, 1)


class LaplacianSharpnessLoss(nn.Module):
    """
    Laplacian-based sharpness loss.
    
    Encourages generated images to have similar high-frequency content (sharpness)
    as the source image. Helps prevent blurry outputs.
    """
    
    def __init__(self):
        super().__init__()
        # Laplacian kernel for edge detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def forward(self, source: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """Match high-frequency content between source and generated."""
        laplacian = self.laplacian.to(source.device, source.dtype)
        
        source_lap = F.conv2d(source, laplacian, padding=1, groups=3)
        gen_lap = F.conv2d(generated, laplacian, padding=1, groups=3)
        
        return F.l1_loss(gen_lap, source_lap)


class GradientMatchingLoss(nn.Module):
    """
    Gradient/Edge matching loss for structure preservation.
    
    Forces generated images to have same edge structure as source,
    critical for preventing mode collapse in image translation.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def get_gradients(self, img: torch.Tensor) -> torch.Tensor:
        """Extract edge gradients from image."""
        # Move filters to same device/dtype
        sobel_x = self.sobel_x.to(img.device, img.dtype)
        sobel_y = self.sobel_y.to(img.device, img.dtype)
        
        # Apply Sobel filters
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=3)
        
        # Gradient magnitude
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    def forward(self, source: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient matching loss between source and generated.
        
        Args:
            source: Source image (structure to preserve)
            generated: Generated image (should have same edges)
        
        Returns:
            L1 loss between gradient magnitudes
        """
        source_grad = self.get_gradients(source)
        gen_grad = self.get_gradients(generated)
        return F.l1_loss(gen_grad, source_grad)


class DomainClassificationLoss(nn.Module):
    """Domain classification loss for discriminator."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        domain_logits: List[torch.Tensor],
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain classification loss."""
        loss = 0.0
        for logits in domain_logits:
            loss += self.ce(logits, domain_labels)
        return loss / len(domain_logits)


class R1Regularization(nn.Module):
    """R1 gradient penalty regularization."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_images: torch.Tensor,
        real_scores: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute R1 regularization with NaN-safe handling."""
        device = real_images.device
        
        try:
            # CRITICAL: Disable autocast for gradient computation
            with torch.amp.autocast('cuda', enabled=False):
                # Convert scores to float32 for stable gradient computation
                real_scores_fp32 = [s.float() for s in real_scores]
                
                grad_outputs = [torch.ones_like(s) for s in real_scores_fp32]
                
                grads = torch.autograd.grad(
                    outputs=real_scores_fp32,
                    inputs=real_images,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                
                # Safety: clamp and clean gradients
                grads = grads.float()
                grads = torch.clamp(grads, -10.0, 10.0)  # Tighter clamp
                grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Compute R1 penalty
                r1 = grads.square().sum([1, 2, 3]).mean()
                
                # Final safety check
                if not torch.isfinite(r1) or r1 > 1000.0:
                    return torch.tensor(0.0, device=device, requires_grad=True)
                
                return r1
                
        except Exception:
            # If gradient computation fails, return 0
            return torch.tensor(0.0, device=device, requires_grad=True)


class MultiStainLoss(nn.Module):
    """
    Combined loss for MultiStain-GAN training.
    
    Aggregates all individual losses with configurable weights.
    """
    
    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_cyc: float = 10.0,
        lambda_sty: float = 1.0,
        lambda_ds: float = 1.0,
        lambda_nce: float = 1.0,
        lambda_perc: float = 0.5,
        lambda_ssim: float = 0.5,
        lambda_l1: float = 10.0,
        lambda_grad: float = 5.0,  # Gradient/edge matching for structure preservation
        lambda_sharp: float = 5.0,  # Sharpness matching to prevent blurry outputs
        nce_temperature: float = 0.07,
        nce_logit_clip: float = 50.0,
        ssim_eps: float = 1e-6,
        lambda_color_hist: float = 5.0,  # NEW: Color histogram matching
        lambda_color_mom: float = 2.0,   # NEW: Color moments matching
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cyc = lambda_cyc
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_nce = lambda_nce
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_sharp = lambda_sharp
        self.lambda_color_hist = lambda_color_hist  # NEW
        self.lambda_color_mom = lambda_color_mom    # NEW
        
        self.adversarial = AdversarialLoss()
        self.cycle = CycleLoss()
        self.style_recon = StyleReconstructionLoss()
        self.style_div = StyleDiversificationLoss()
        self.nce = PatchNCELoss(temperature=nce_temperature, logit_clip=nce_logit_clip)
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss(eps=ssim_eps)
        self.gradient = GradientMatchingLoss()  # Edge/structure matching
        self.sharpness = LaplacianSharpnessLoss()  # Prevent blurry outputs
        self.domain_cls = DomainClassificationLoss()
        self.r1 = R1Regularization()
        # NEW: Color-specific losses to prevent color collapse
        self.color_hist = ColorHistogramLoss(num_bins=64)  # Histogram matching
        self.color_moments = ColorMomentsLoss()  # Mean/std/skewness matching
    
    def generator_loss(
        self,
        fake_scores: List[torch.Tensor],
        source: torch.Tensor,
        reconstructed: torch.Tensor,
        style_target: torch.Tensor,
        style_generated: torch.Tensor,
        source_features: Optional[List[torch.Tensor]] = None,
        target_features: Optional[List[torch.Tensor]] = None,
        fake: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        fake_domain_logits: Optional[List[torch.Tensor]] = None,
        target_domain: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses."""
        losses = {}
        device = source.device
        
        # Adversarial (with NaN protection)
        adv_loss = self.adversarial([], fake_scores, mode='generator')
        if not torch.isfinite(adv_loss):
            adv_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses['adv'] = self.lambda_adv * adv_loss
        
        # CRITICAL: Auxiliary domain classification on FAKE images
        # Forces generator to produce domain-specific outputs instead of ignoring domain labels
        if fake_domain_logits is not None and target_domain is not None:
            domain_loss = self.domain_cls(fake_domain_logits, target_domain)
            losses['domain'] = domain_loss  # Weight 1.0 - important for domain awareness
        
        # Cycle consistency
        losses['cyc'] = self.lambda_cyc * self.cycle(source, reconstructed)
        
        # Style reconstruction
        losses['sty'] = self.lambda_sty * self.style_recon(style_target, style_generated)
        
        # Contrastive NCE (compare source encoder features vs fake encoder features)
        if source_features is not None and target_features is not None:
            losses['nce'] = self.lambda_nce * self.nce(source_features, target_features)
        
        # CRITICAL: Gradient/Edge matching - forces structure preservation from source!
        # This is KEY to preventing mode collapse - fake must have same edges as source
        if fake is not None:
            losses['grad'] = self.lambda_grad * self.gradient(source, fake)
            # Sharpness matching - prevent blurry outputs
            losses['sharp'] = self.lambda_sharp * self.sharpness(source, fake)
        
        # Perceptual, SSIM & L1 reconstruction (if pairised data available)
        if target is not None and fake is not None:
            losses['perc'] = self.lambda_perc * self.perceptual(fake, target)
            losses['ssim'] = self.lambda_ssim * self.ssim(fake, target)
            # Direct L1 reconstruction - for color matching with target
            losses['l1'] = self.lambda_l1 * F.l1_loss(fake, target)
            
            # NEW: Color-specific losses - CRITICAL for preventing color collapse
            # Color histogram matching - forces similar color distribution to target domain
            losses['color_hist'] = self.lambda_color_hist * self.color_hist(fake, target)
            # Color moments matching - mean, std, skewness per channel
            losses['color_mom'] = self.lambda_color_mom * self.color_moments(fake, target)
        
        # Compute total with NaN protection for each component
        total = torch.tensor(0.0, device=device, requires_grad=True)
        for k, v in losses.items():
            if torch.isfinite(v):
                total = total + v
            else:
                # Replace NaN/Inf with zero and log warning
                losses[k] = torch.tensor(0.0, device=device, requires_grad=True)
        losses['total'] = total
        return losses
    
    def discriminator_loss(
        self,
        real_scores: List[torch.Tensor],
        fake_scores: List[torch.Tensor],
        real_domain_logits: List[torch.Tensor],
        domain_labels: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        r1_gamma: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses."""
        losses = {}
        
        # Adversarial
        losses['adv'] = self.adversarial(real_scores, fake_scores, mode='discriminator')
        
        # Domain classification
        losses['domain'] = self.domain_cls(real_domain_logits, domain_labels)
        
        # R1 regularization (optional)
        if real_images is not None and real_images.requires_grad:
            losses['r1'] = r1_gamma * self.r1(real_images, real_scores)
        
        losses['total'] = sum(losses.values())
        return losses
