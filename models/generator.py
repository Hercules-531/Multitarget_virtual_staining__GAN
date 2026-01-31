"""
Generator with Pretrained Encoder and Efficient ViT

Combines pretrained ResNet encoder with efficient ViT blocks and AdaIN for style injection.
Includes gradient checkpointing for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List

import torchvision.models as models


class AdaIN(nn.Module):
    """Adaptive Instance Normalization for style injection."""
    
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        # Initialize to have stronger modulation from the start
        nn.init.zeros_(self.fc.bias)
        # Scale weights so gamma starts around 1.0 (stronger modulation)
        nn.init.normal_(self.fc.weight, 0, 0.1)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = h.chunk(2, dim=1)
        # Use exp(gamma) for always-positive scaling with stronger effect
        # This ensures style always modulates the output significantly
        scale = torch.exp(gamma.clamp(-2, 2))  # Clamp for stability, range [0.13, 7.4]
        return scale * self.norm(x) + beta


class ResBlock(nn.Module):
    """Residual block with AdaIN style injection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int = 64,
        upsample: bool = False,
    ):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.adain1 = AdaIN(out_channels, style_dim)
        self.adain2 = AdaIN(out_channels, style_dim)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            residual = F.interpolate(residual, scale_factor=2, mode='bilinear', align_corners=False)
        
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.adain2(out, style)
        
        out = out + self.shortcut(residual)
        return self.activation(out)


class EfficientViTBlock(nn.Module):
    """Efficient Vision Transformer block with attention dropout."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, attn_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1),
        )
        # Learnable scale for residual - helps with gradient flow
        self.gamma1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-attention with learnable residual scale
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = self.attn_drop(attn_out)
        x_flat = x_flat + self.gamma1 * attn_out
        
        # MLP with learnable residual scale
        x_flat = x_flat + self.gamma2 * self.mlp(self.norm2(x_flat))
        
        # Reshape back
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class ResNetEncoder(nn.Module):
    """Pretrained ResNet encoder for feature extraction."""
    
    def __init__(self, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512
        
        # Channel dimensions for skip connections
        self.channels = [64, 64, 128, 256, 512]
        
        # ImageNet normalization buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns multi-scale features for skip connections."""
        x = self._normalize(x)
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels, /2
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 64 channels, /4
        
        x = self.layer2(x)
        features.append(x)  # 128 channels, /8
        
        x = self.layer3(x)
        features.append(x)  # 256 channels, /16
        
        x = self.layer4(x)
        features.append(x)  # 512 channels, /32
        
        return features


class Generator(nn.Module):
    """
    Generator with Pretrained ResNet Encoder and Efficient ViT.
    
    Key optimizations:
    - Pretrained ResNet-34 encoder for better feature extraction
    - Efficient ViT blocks with PyTorch native MultiheadAttention
    - Gradient checkpointing for memory efficiency (optional)
    - Skip connections from encoder to decoder
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        style_dim: int = 64,
        enc_channels: List[int] = [64, 128, 256, 512],
        num_res_blocks: int = 6,
        vit_blocks: int = 4,
        vit_heads: int = 8,
        use_pretrained: bool = True,
        use_checkpoint: bool = False,  # Gradient checkpointing
    ):
        super().__init__()
        self.image_size = image_size
        self.style_dim = style_dim
        self.use_checkpoint = use_checkpoint
        
        # Pretrained encoder
        self.encoder = ResNetEncoder(pretrained=use_pretrained)
        encoder_out_dim = 512
        
        # Project to working dimension
        self.enc_proj = nn.Conv2d(encoder_out_dim, enc_channels[-1], 1)
        
        # Efficient ViT blocks for global context
        self.vit_blocks = nn.ModuleList([
            EfficientViTBlock(enc_channels[-1], vit_heads)
            for _ in range(vit_blocks)
        ])
        
        # Residual blocks with AdaIN
        self.res_blocks = nn.ModuleList([
            ResBlock(enc_channels[-1], enc_channels[-1], style_dim)
            for _ in range(num_res_blocks)
        ])
        
        # Decoder with skip connections
        # dec_channels = [512, 256, 128, 64]
        # After decoder block 0: 512->256 (output 256)
        # After decoder block 1: 256->128 (output 128)
        # After decoder block 2: 128->64  (output 64)
        dec_channels = list(reversed(enc_channels))  # [512, 256, 128, 64]
        self.decoder = nn.ModuleList()
        for i in range(len(dec_channels) - 1):
            self.decoder.append(
                ResBlock(dec_channels[i], dec_channels[i + 1], style_dim, upsample=True)
            )
        
        # Skip connection projections (from encoder to decoder OUTPUT channels)
        # After decoder[0] (outputs 256ch): add skip from enc layer3 (256ch) -> 256ch
        # After decoder[1] (outputs 128ch): add skip from enc layer2 (128ch) -> 128ch
        # After decoder[2] (outputs 64ch):  add skip from enc layer1 (64ch)  -> 64ch
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(256, 256, 1),  # decoder[0] output is 256ch
            nn.Conv2d(128, 128, 1),  # decoder[1] output is 128ch
            nn.Conv2d(64, 64, 1),    # decoder[2] output is 64ch
        ])
        
        # Final upsampling stages to reach target resolution
        # After decoder: 8 -> 16 -> 32 -> 64 (3 decoder blocks)
        # Need to go: 64 -> 128 -> 256 (2 more upsample stages)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, 1, 1),
            nn.InstanceNorm2d(dec_channels[-1]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, 1, 1),
            nn.InstanceNorm2d(dec_channels[-1]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.to_rgb = nn.Sequential(
            nn.Conv2d(dec_channels[-1], out_channels, 7, 1, 3),
            nn.Tanh(),
        )

    
    def _run_vit(self, x: torch.Tensor) -> torch.Tensor:
        """Run ViT blocks with optional gradient checkpointing."""
        for vit in self.vit_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(vit, x, use_reentrant=False)
            else:
                x = vit(x)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W) in range [-1, 1]
            style: Style code (B, style_dim)
            return_features: If True, return multi-scale encoder features for NCE loss
        
        Returns:
            Generated image (B, C, H, W) in range [-1, 1]
            If return_features: also returns list of encoder features at multiple scales
        """
        # Encode with pretrained ResNet
        enc_features = self.encoder(x)
        
        # Get bottleneck features
        out = self.enc_proj(enc_features[-1])
        
        # ViT for global context
        out = self._run_vit(out)
        
        # Residual blocks with style injection
        for res_block in self.res_blocks:
            if self.use_checkpoint and self.training:
                out = checkpoint(lambda o, s: res_block(o, s), out, style, use_reentrant=False)
            else:
                out = res_block(out, style)
        
        # Decoder with skip connections - VERY strong skip weights preserve spatial structure
        skip_idx = [3, 2, 1]  # Indices for 256, 128, 64 channel features
        for i, dec in enumerate(self.decoder):
            out = dec(out, style)
            if i < len(self.skip_projs):
                skip = self.skip_projs[i](enc_features[skip_idx[i]])
                out = out + skip  # Full skip connections (1.0) - critical for structure
        
        # Final upsampling: 64 -> 128 -> 256
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.to_rgb(out)
        
        if return_features:
            # Return multi-scale encoder features (not post-generation features)
            # These are used for NCE loss to compare source vs fake structure
            return out, enc_features
        return out
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent features."""
        enc_features = self.encoder(x)
        out = self.enc_proj(enc_features[-1])
        out = self._run_vit(out)
        return out
