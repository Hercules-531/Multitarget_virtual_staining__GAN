"""
Mapping Network

Maps random latent codes to domain-specific style codes.
"""

import torch
import torch.nn as nn
from typing import List


class MappingNetwork(nn.Module):
    """
    Mapping Network for random style code generation.
    
    Maps a random latent vector z to domain-specific style codes,
    enabling diverse image generation within each target domain.
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 64,
        num_domains: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_domains = num_domains
        
        # Shared MLP layers
        shared_layers = []
        shared_layers.append(nn.Linear(latent_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Domain-specific output heads
        self.domain_heads = nn.ModuleList([
            nn.Linear(hidden_dim, style_dim)
            for _ in range(num_domains)
        ])
    
    def forward(
        self,
        z: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map latent code to style code for specified domain.
        
        Args:
            z: Latent code (B, latent_dim)
            domain: Domain indices (B,)
        
        Returns:
            Style code (B, style_dim)
        """
        # Shared transformation
        h = self.shared(z)
        
        # Get domain-specific style codes
        batch_size = z.size(0)
        style = torch.zeros(batch_size, self.style_dim, device=z.device)
        
        for i in range(self.num_domains):
            mask = (domain == i)
            if mask.any():
                style[mask] = self.domain_heads[i](h[mask])
        
        return style
    
    def forward_all_domains(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map latent code to style codes for all domains.
        
        Args:
            z: Latent code (B, latent_dim)
        
        Returns:
            Style codes (B, num_domains, style_dim)
        """
        # Shared transformation
        h = self.shared(z)
        
        # Get all domain style codes
        styles = torch.stack([
            head(h) for head in self.domain_heads
        ], dim=1)
        
        return styles
    
    def sample(
        self,
        batch_size: int,
        domain: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample random style codes for specified domains.
        
        Args:
            batch_size: Number of samples
            domain: Domain indices (B,) or single int
            device: Device to create tensors on
        
        Returns:
            Style codes (B, style_dim)
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        if isinstance(domain, int):
            domain = torch.full((batch_size,), domain, dtype=torch.long, device=device)
        return self.forward(z, domain)
