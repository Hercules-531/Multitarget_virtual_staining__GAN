"""
Multi-Scale Discriminator

PatchGAN discriminator operating at multiple scales for better realism.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def spectral_norm(module: nn.Module, use: bool = True) -> nn.Module:
    """Apply spectral normalization if requested."""
    if use:
        return nn.utils.spectral_norm(module)
    return module


class MinibatchStdDev(nn.Module):
    """
    Minibatch Standard Deviation layer.
    
    Computes the standard deviation across the batch for each spatial location,
    then averages across spatial dimensions and appends as an extra channel.
    This explicitly penalizes lack of variance in generated batches,
    helping prevent mode collapse.
    """
    
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Adjust group size if batch is smaller
        G = min(self.group_size, B)
        if B % G != 0:
            G = 1  # Fall back to computing std across whole batch
        
        # Reshape: (B, C, H, W) -> (G, B//G, C, H, W)
        y = x.view(G, B // G, C, H, W)
        
        # Compute std across batch dimension within each group
        y = y - y.mean(dim=0, keepdim=True)  # Center
        y = (y ** 2).mean(dim=0)  # Variance
        y = (y + 1e-8).sqrt()  # Std with epsilon for stability
        
        # Average over channels and spatial dims: (B//G, C, H, W) -> (B//G, 1, 1, 1)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        
        # Tile to match batch size: (B//G, 1, 1, 1) -> (B, 1, H, W)
        y = y.repeat(G, 1, H, W)
        
        # Concatenate as extra channel
        return torch.cat([x, y], dim=1)


class DiscriminatorBlock(nn.Module):
    """Single discriminator block with spectral normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1),
            use_spectral_norm
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Skip InstanceNorm if spatial size is too small (<=2)
        # InstanceNorm requires more than 1 spatial element
        if x.size(-1) > 2 and x.size(-2) > 2:
            x = self.norm(x)
        x = self.activation(x)
        return x


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    Outputs real/fake score for each patch in the image.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        num_domains: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_domains = num_domains
        
        # Initial layer (no normalization)
        layers = [
            spectral_norm(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1),
                use_spectral_norm
            ),
            nn.LeakyReLU(0.2),
        ]
        
        # Intermediate layers
        channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(
                DiscriminatorBlock(channels, out_channels, stride, use_spectral_norm)
            )
            channels = out_channels
        
        self.main = nn.Sequential(*layers)
        
        # Minibatch StdDev - CRITICAL for preventing mode collapse!
        # Explicitly penalizes generator if all outputs look the same
        self.minibatch_std = MinibatchStdDev(group_size=4)
        
        # Output heads - use 3x3 kernel, +1 channel for minibatch std
        self.real_fake = spectral_norm(
            nn.Conv2d(channels + 1, 1, 3, 1, 1),  # +1 for minibatch std channel
            use_spectral_norm
        )
        
        # Domain classification head
        self.domain_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_domains),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            real_fake: Real/fake scores (B, 1, H', W')
            domain_logits: Domain classification logits (B, num_domains)
        """
        features = self.main(x)
        # Apply minibatch std before final classification
        features_with_std = self.minibatch_std(features)
        real_fake = self.real_fake(features_with_std)
        domain_logits = self.domain_cls(features)  # Domain cls uses original features
        return real_fake, domain_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features for feature matching loss."""
        return self.main(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator.
    
    Operates at multiple scales for better gradient flow and realism.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        num_domains: int = 4,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = num_scales
        
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                PatchDiscriminator(
                    in_channels,
                    base_channels,
                    num_layers,
                    num_domains,
                    use_spectral_norm,
                )
            )
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            real_fake_list: List of real/fake scores at each scale
            domain_logits_list: List of domain logits at each scale
        """
        real_fake_list = []
        domain_logits_list = []
        
        for i, disc in enumerate(self.discriminators):
            real_fake, domain_logits = disc(x)
            real_fake_list.append(real_fake)
            domain_logits_list.append(domain_logits)
            
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return real_fake_list, domain_logits_list
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get features at all scales."""
        features = []
        for i, disc in enumerate(self.discriminators):
            features.append(disc.get_features(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        return features
