"""
Style Encoder with Pretrained ResNet Backbone

Extracts domain-specific style codes from reference images using pretrained features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision.models as models


class StyleEncoder(nn.Module):
    """
    Style Encoder using pretrained ResNet-18 backbone.
    
    Pretrained features significantly improve style extraction quality
    by leveraging learned representations from ImageNet.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        style_dim: int = 64,
        num_domains: int = 4,
        pretrained: bool = True,
        freeze_layers: int = 2,  # Freeze first N layers for stable training
    ):
        super().__init__()
        self.style_dim = style_dim
        self.num_domains = num_domains
        
        # Load pretrained ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Extract encoder layers (remove FC and avgpool)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Freeze early layers for stable training
        if pretrained and freeze_layers > 0:
            frozen = [self.conv1, self.bn1, self.layer1, self.layer2][:freeze_layers]
            for layer in frozen:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension from ResNet-18
        feat_dim = 512
        
        # Shared feature refinement
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # Domain-specific heads
        self.domain_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, style_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(style_dim * 2, style_dim),
            )
            for _ in range(num_domains)
        ])
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using ResNet backbone."""
        x = self._normalize(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x).flatten(1)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract style code from reference image for specified domain.
        
        Args:
            x: Reference image (B, C, H, W) in range [-1, 1]
            domain: Domain indices (B,)
        
        Returns:
            Style code (B, style_dim)
        """
        h = self._encode(x)
        h = self.shared(h)
        
        batch_size = x.size(0)
        # Use h.dtype to match AMP precision (may be FP16 in autocast)
        style = torch.zeros(batch_size, self.style_dim, device=x.device, dtype=h.dtype)
        
        for i in range(self.num_domains):
            mask = (domain == i)
            if mask.any():
                style[mask] = self.domain_heads[i](h[mask])
        
        return style
    
    def forward_all_domains(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract style codes for all domains.
        
        Args:
            x: Reference image (B, C, H, W)
        
        Returns:
            Style codes (B, num_domains, style_dim)
        """
        h = self._encode(x)
        h = self.shared(h)
        
        styles = torch.stack([
            head(h) for head in self.domain_heads
        ], dim=1)
        
        return styles
