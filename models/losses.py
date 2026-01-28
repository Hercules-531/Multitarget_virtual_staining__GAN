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
    """Hinge adversarial loss (more stable than LSGAN)."""
    
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
        Compute adversarial loss.
        
        Args:
            real_scores: List of real image scores from multi-scale D
            fake_scores: List of fake image scores from multi-scale D
            mode: 'discriminator' or 'generator'
        
        Returns:
            Loss value
        """
        loss = 0.0
        
        if self.loss_type == "hinge":
            # Hinge loss - more stable for GAN training
            if mode == "discriminator":
                for real, fake in zip(real_scores, fake_scores):
                    # Real should be > 1, fake should be < -1
                    loss += torch.mean(F.relu(1.0 - real))
                    loss += torch.mean(F.relu(1.0 + fake))
            else:  # generator
                for fake in fake_scores:
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


class StyleDiversificationLoss(nn.Module):
    """Style diversification loss to encourage diverse outputs."""
    
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
        
        Maximizes image difference relative to style difference.
        """
        style_diff = self.l1(style1, style2)
        image_diff = self.l1(image1, image2)
        
        # We want image_diff to be large when style_diff is large
        # Maximize image_diff / (style_diff + eps)
        # Equivalent to minimizing -image_diff
        return -image_diff


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
        # Normalize to [0, 1]
        generated = generated.clamp(-1.0, 1.0)
        target = target.clamp(-1.0, 1.0)
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(generated, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(generated ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(generated * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        sigma1_sq = sigma1_sq.clamp(min=0.0)
        sigma2_sq = sigma2_sq.clamp(min=0.0)
        
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        denom = denom + self.eps
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denom
        
        return 1 - ssim_map.mean()


class PatchNCELoss(nn.Module):
    """
    Patch-wise Contrastive Loss (NCE).
    
    Maximizes mutual information between corresponding patches.
    """
    
    def __init__(
        self,
        nce_layers: List[int] = [0, 1],
        num_patches: int = 256,
        temperature: float = 0.07,
        logit_clip: float = 50.0,
    ):
        super().__init__()
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.temperature = max(temperature, 1e-4)
        self.logit_clip = logit_clip
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
        # MLP heads for each layer
        self.mlps = nn.ModuleList()
    
    def _init_mlps(self, features: List[torch.Tensor]):
        """Initialize MLP heads based on feature dimensions."""
        if len(self.mlps) == 0:
            for feat in features:
                nc = feat.shape[1]
                self.mlps.append(
                    nn.Sequential(
                        nn.Linear(nc, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                    ).to(feat.device)
                )
    
    def forward(
        self,
        source_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute PatchNCE loss.
        
        Args:
            source_features: Features from source image encoder
            target_features: Features from generated image encoder
        
        Returns:
            NCE loss value
        """
        self._init_mlps(source_features)
        
        total_loss = 0.0
        
        for i, (feat_s, feat_t) in enumerate(zip(source_features, target_features)):
            B, C, H, W = feat_s.shape
            
            # Sample random patches
            num_patches = min(self.num_patches, H * W)
            indices = torch.randperm(H * W)[:num_patches]
            
            # Reshape to (B, C, H*W)
            feat_s = feat_s.flatten(2)
            feat_t = feat_t.flatten(2)
            
            # Sample patches
            feat_s = feat_s[:, :, indices]  # (B, C, num_patches)
            feat_t = feat_t[:, :, indices]
            
            # Project through MLP
            feat_s = feat_s.permute(0, 2, 1).reshape(-1, C)  # (B*num_patches, C)
            feat_t = feat_t.permute(0, 2, 1).reshape(-1, C)
            
            if i < len(self.mlps):
                feat_s = self.mlps[i](feat_s)
                feat_t = self.mlps[i](feat_t)
            
            # Normalize
            feat_s = F.normalize(feat_s, dim=1)
            feat_t = F.normalize(feat_t, dim=1)
            
            # Compute similarities
            logits = torch.mm(feat_t, feat_s.t()) / self.temperature
            logits = logits - logits.max(dim=1, keepdim=True).values
            if self.logit_clip is not None:
                logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
            
            # Labels: positive pairs are on diagonal
            labels = torch.arange(logits.size(0), device=logits.device)
            
            loss = self.cross_entropy(logits, labels)
            total_loss += loss.mean()
        
        return total_loss / len(source_features)


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
        try:
            grad_outputs = [torch.ones_like(s) for s in real_scores]
            
            grads = torch.autograd.grad(
                outputs=real_scores,
                inputs=real_images,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            # Clamp gradients to prevent explosion
            grads = torch.clamp(grads, -100.0, 100.0)
            
            # Replace any NaN/Inf with 0
            grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            
            r1 = grads.square().sum([1, 2, 3]).mean()
            
            # Final safety check
            if not torch.isfinite(r1):
                return torch.tensor(0.0, device=real_images.device, requires_grad=True)
            
            return r1
        except Exception:
            # If gradient computation fails, return 0
            return torch.tensor(0.0, device=real_images.device, requires_grad=True)


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
        nce_temperature: float = 0.07,
        nce_logit_clip: float = 50.0,
        ssim_eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cyc = lambda_cyc
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_nce = lambda_nce
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        
        self.adversarial = AdversarialLoss()
        self.cycle = CycleLoss()
        self.style_recon = StyleReconstructionLoss()
        self.style_div = StyleDiversificationLoss()
        self.nce = PatchNCELoss(temperature=nce_temperature, logit_clip=nce_logit_clip)
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss(eps=ssim_eps)
        self.domain_cls = DomainClassificationLoss()
        self.r1 = R1Regularization()
    
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
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses."""
        losses = {}
        
        # Adversarial
        losses['adv'] = self.lambda_adv * self.adversarial([], fake_scores, mode='generator')
        
        # Cycle consistency
        losses['cyc'] = self.lambda_cyc * self.cycle(source, reconstructed)
        
        # Style reconstruction
        losses['sty'] = self.lambda_sty * self.style_recon(style_target, style_generated)
        
        # Contrastive NCE
        if source_features is not None and target_features is not None:
            losses['nce'] = self.lambda_nce * self.nce(source_features, target_features)
        
        # Perceptual & SSIM (if paired data available)
        if target is not None and fake is not None:
            losses['perc'] = self.lambda_perc * self.perceptual(fake, target)
            losses['ssim'] = self.lambda_ssim * self.ssim(fake, target)
        
        losses['total'] = sum(losses.values())
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
