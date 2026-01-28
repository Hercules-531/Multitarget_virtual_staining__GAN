"""
Vision Transformer Blocks for Generator

Provides global context and long-range dependencies for better structural preservation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP block for ViT."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    """Single Vision Transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for global context.
    
    Takes feature maps from CNN encoder and applies self-attention.
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        
        # Position embedding (learnable)
        self.pos_embed = None  # Will be initialized on first forward
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(dim)
    
    def _init_pos_embed(self, H: int, W: int, device: torch.device):
        """Initialize positional embeddings."""
        num_patches = H * W
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.dim, device=device)
        )
        # Initialize with sinusoidal encoding
        position = torch.arange(num_patches, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) * 
            (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(1, num_patches, self.dim, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data = pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            Transformed features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add positional embedding
        if self.pos_embed is None or self.pos_embed.size(1) != H * W:
            self._init_pos_embed(H, W, x.device)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for feature maps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Create 2D positional encoding
        y_pos = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=x.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize to [0, 1]
        y_pos = y_pos.float() / H
        x_pos = x_pos.float() / W
        
        # Create sinusoidal encoding
        dim_t = torch.arange(0, C // 4, device=x.device).float()
        dim_t = 10000 ** (2 * dim_t / C)
        
        pe_y = y_pos.unsqueeze(-1) / dim_t
        pe_x = x_pos.unsqueeze(-1) / dim_t
        
        pe = torch.cat([
            pe_y.sin(), pe_y.cos(),
            pe_x.sin(), pe_x.cos()
        ], dim=-1)  # (H, W, C)
        
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return x + pe[:, :C]
