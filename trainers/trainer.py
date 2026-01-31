"""
Training Loop for MultiStain-GAN

Handles multi-domain virtual staining training with all loss components.
Includes optimizations: mixed precision, gradient accumulation, EMA, and more.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import Generator, MultiScaleDiscriminator, StyleEncoder, MappingNetwork
from models.losses import MultiStainLoss


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    """
    MultiStain-GAN Trainer.
    
    Optimizations:
    - Mixed precision (FP16) training
    - Gradient accumulation for larger effective batch size
    - EMA for stable generation
    - AdamW optimizer with weight decay
    - Cosine annealing LR schedule
    """
    
    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Enable cuDNN benchmark for faster convolutions
        if config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        # Model configuration
        model_cfg = config.get('model', {})
        train_cfg = config.get('training', {})
        gen_cfg = model_cfg.get('generator', {})
        
        # Initialize models with pretrained options
        self.generator = Generator(
            image_size=model_cfg.get('image_size', 256),
            style_dim=model_cfg.get('style_dim', 64),
            enc_channels=gen_cfg.get('enc_channels', [64, 128, 256, 512]),
            num_res_blocks=gen_cfg.get('num_res_blocks', 6),
            vit_blocks=gen_cfg.get('vit_blocks', 4),
            vit_heads=gen_cfg.get('vit_heads', 8),
            use_pretrained=gen_cfg.get('use_pretrained', True),
            use_checkpoint=gen_cfg.get('use_checkpoint', True),
        ).to(self.device)
        
        self.discriminator = MultiScaleDiscriminator(
            base_channels=model_cfg.get('discriminator', {}).get('base_channels', 64),
            num_layers=model_cfg.get('discriminator', {}).get('num_layers', 4),
            num_domains=model_cfg.get('num_domains', 4),
            num_scales=model_cfg.get('discriminator', {}).get('num_scales', 3),
        ).to(self.device)
        
        style_cfg = model_cfg.get('style_encoder', {})
        self.style_encoder = StyleEncoder(
            image_size=model_cfg.get('image_size', 256),
            style_dim=model_cfg.get('style_dim', 64),
            num_domains=model_cfg.get('num_domains', 4),
            pretrained=style_cfg.get('pretrained', True),
            freeze_layers=style_cfg.get('freeze_layers', 2),
        ).to(self.device)
        
        self.mapping_network = MappingNetwork(
            latent_dim=model_cfg.get('latent_dim', 16),
            style_dim=model_cfg.get('style_dim', 64),
            num_domains=model_cfg.get('num_domains', 4),
            num_layers=model_cfg.get('mapping_network', {}).get('num_layers', 4),
            hidden_dim=model_cfg.get('mapping_network', {}).get('hidden_dim', 512),
        ).to(self.device)
        
        # NOTE: torch.compile disabled - causes issues with instance_norm and symbolic shapes
        # If you want to try it, uncomment below (may need PyTorch 2.2+)
        # if config.get('compile_model', False) and hasattr(torch, 'compile'):
        #     print("Compiling models with torch.compile()...")
        #     self.generator = torch.compile(self.generator, mode='reduce-overhead')
        #     self.discriminator = torch.compile(self.discriminator, mode='reduce-overhead')
        
        # Optimizers (AdamW for better weight decay)
        opt_cfg = train_cfg.get('optimizer', {})
        lr_g = opt_cfg.get('lr_generator', 1e-4)
        lr_d = opt_cfg.get('lr_discriminator', 4e-4)
        lr_style = opt_cfg.get('lr_style_encoder', lr_g * 0.1)  # Lower LR for unfrozen style encoder
        betas = (opt_cfg.get('beta1', 0.0), opt_cfg.get('beta2', 0.99))
        weight_decay = opt_cfg.get('weight_decay', 1e-2)
        
        OptClass = torch.optim.AdamW if opt_cfg.get('type', 'AdamW') == 'AdamW' else torch.optim.Adam
        
        # Use parameter groups with different learning rates
        # Style encoder gets lower LR since it's now unfrozen
        self.optimizer_g = OptClass(
            [
                {'params': self.generator.parameters(), 'lr': lr_g},
                {'params': self.mapping_network.parameters(), 'lr': lr_g},
                {'params': self.style_encoder.parameters(), 'lr': lr_style},  # Lower LR
            ],
            betas=betas,
            weight_decay=weight_decay,
        )
        
        self.optimizer_d = OptClass(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=betas,
            weight_decay=weight_decay,
        )
        
        # Schedulers (Cosine Annealing)
        sched_cfg = train_cfg.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'CosineAnnealingLR')
        
        if sched_type == 'CosineAnnealingLR':
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g,
                T_max=sched_cfg.get('T_max', 150),
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d,
                T_max=sched_cfg.get('T_max', 150),
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
        else:
            self.scheduler_g = torch.optim.lr_scheduler.StepLR(
                self.optimizer_g,
                step_size=sched_cfg.get('step_size', 50),
                gamma=sched_cfg.get('gamma', 0.5),
            )
            self.scheduler_d = torch.optim.lr_scheduler.StepLR(
                self.optimizer_d,
                step_size=sched_cfg.get('step_size', 50),
                gamma=sched_cfg.get('gamma', 0.5),
            )
        
        # Loss functions
        loss_cfg = train_cfg.get('losses', {})
        self.criterion = MultiStainLoss(
            lambda_adv=loss_cfg.get('adversarial', 1.0),
            lambda_cyc=loss_cfg.get('cycle', 10.0),
            lambda_sty=loss_cfg.get('style_reconstruction', 1.0),
            lambda_ds=loss_cfg.get('style_diversification', 1.0),
            lambda_nce=loss_cfg.get('contrastive_nce', 1.0),
            lambda_perc=loss_cfg.get('perceptual', 0.5),
            lambda_ssim=loss_cfg.get('ssim', 0.5),
            lambda_l1=loss_cfg.get('l1_reconstruction', 10.0),
            lambda_grad=loss_cfg.get('gradient_matching', 5.0),  # Edge/structure preservation
            lambda_sharp=loss_cfg.get('sharpness_matching', 5.0),  # Prevent blurry outputs
            nce_temperature=loss_cfg.get('nce_temperature', 0.07),
            nce_logit_clip=loss_cfg.get('nce_logit_clip', 50.0),
            ssim_eps=loss_cfg.get('ssim_eps', 1e-6),
            # NEW: Color-specific losses for color collapse prevention
            lambda_color_hist=loss_cfg.get('color_histogram', 5.0),  # Histogram matching
            lambda_color_mom=loss_cfg.get('color_moments', 2.0),     # Color moments matching
        )
        
        # Mixed precision
        self.use_amp = config.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping for training stability
        self.grad_clip_max_norm = train_cfg.get('grad_clip_max_norm', 1.0)
        self.r1_interval = train_cfg.get('r1_interval', 16)
        
        # Gradient accumulation
        self.accumulate_grad = train_cfg.get('accumulate_grad', 1)
        
        # EMA for generator
        ema_decay = train_cfg.get('ema_decay', 0.999)
        self.ema_g = EMA(self.generator, decay=ema_decay) if ema_decay > 0 else None
        
        # Logging
        log_cfg = config.get('logging', {})
        self.log_dir = Path(log_cfg.get('log_dir', './logs'))
        self.checkpoint_dir = Path(log_cfg.get('checkpoint_dir', './checkpoints'))
        self.sample_dir = Path(log_cfg.get('sample_dir', './samples'))
        
        for d in [self.log_dir, self.checkpoint_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        self.log_every = log_cfg.get('log_every', 100)
        self.save_every = log_cfg.get('save_every', 5000)
        self.sample_every = log_cfg.get('sample_every', 1000)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_fid = float('inf')
        
        # Print model info
        self._print_model_info()
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _print_model_info(self):
        """Print model parameter counts."""
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{'='*50}")
        print("Model Parameters:")
        print(f"  Generator:       {count_params(self.generator):,}")
        print(f"  Discriminator:   {count_params(self.discriminator):,}")
        print(f"  Style Encoder:   {count_params(self.style_encoder):,}")
        print(f"  Mapping Network: {count_params(self.mapping_network):,}")
        print(f"  Total:           {count_params(self.generator) + count_params(self.discriminator) + count_params(self.style_encoder) + count_params(self.mapping_network):,}")
        print(f"{'='*50}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.accumulate_grad}")
        print(f"EMA: {self.ema_g is not None}")
        print(f"{'='*50}\n")
    
    def train_step(self, batch: Dict[str, torch.Tensor], accum_step: int) -> Dict[str, float]:
        """Single training step with gradient accumulation support."""
        source = batch['source'].to(self.device, non_blocking=True)
        target_domain = batch['target_domain'].to(self.device, non_blocking=True).long()
        reference = batch['reference'].to(self.device, non_blocking=True)
        target = batch.get('target')
        if target is not None:
            target = target.to(self.device, non_blocking=True)
        
        # Check for NaN inputs (data corruption)
        if torch.isnan(source).any() or torch.isnan(reference).any():
            print("WARNING: NaN detected in input data, skipping batch")
            return {'d_total': 0.0, 'g_total': 0.0}
        
        batch_size = source.size(0)
        is_last_accum = (accum_step + 1) == self.accumulate_grad
        
        # Instance noise for discriminator regularization (helps prevent mode collapse)
        noise_std = self.config['training'].get('noise_std', 0.05)  # Reduced from 0.1
        # Decay noise over training (annealed instance noise)
        noise_decay = max(0.0, 1.0 - self.global_step / 50000)
        current_noise_std = noise_std * noise_decay
        
        # Color jitter for D inputs only (forces D to be color-agnostic, pushing G to diversify)
        # Apply to ~30% of batches for regularization without destabilizing training
        use_color_jitter = (self.global_step % 3 == 0)  # Every 3rd step
        
        # ========== Train Discriminator ==========
        z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
        style_random = self.mapping_network(z, target_domain)
        
        with torch.no_grad():
            fake = self.generator(source, style_random)
        
        real = reference.clone()
        do_r1 = (self.global_step % self.r1_interval == 0) and (accum_step == 0)
        real.requires_grad_(do_r1)
        
        # Add instance noise to D inputs (regularization to prevent mode collapse)
        # Always apply during training (train_step is only called during training)
        if current_noise_std > 0:
            real_noisy = real + current_noise_std * torch.randn_like(real)
            fake_noisy = fake.detach() + current_noise_std * torch.randn_like(fake)
        else:
            real_noisy = real
            fake_noisy = fake.detach()
        
        # Apply color jitter to D inputs only (not G) to force color diversity
        # This makes D color-agnostic, preventing G from producing uniform colors
        if use_color_jitter:
            # Random color adjustments (small to avoid instability)
            brightness = 0.1 * (torch.rand(1, device=self.device).item() - 0.5)
            contrast = 1.0 + 0.1 * (torch.rand(1, device=self.device).item() - 0.5)
            
            # Apply same jitter to both real and fake for consistency
            real_noisy = (real_noisy * contrast + brightness).clamp(-1, 1)
            fake_noisy = (fake_noisy * contrast + brightness).clamp(-1, 1)
        
        with autocast(device_type='cuda', enabled=self.use_amp):
            real_scores, real_domain_logits = self.discriminator(real_noisy)
            fake_scores, _ = self.discriminator(fake_noisy)
        
        # Cast to float32 for stable loss computation
        real_scores_fp32 = [s.float() for s in real_scores]
        fake_scores_fp32 = [s.float() for s in fake_scores]
        
        d_losses = self.criterion.discriminator_loss(
            real_scores_fp32, fake_scores_fp32,
            [logit.float() for logit in real_domain_logits], target_domain,
            real_images=real if do_r1 else None,
            r1_gamma=self.config['training'].get('r1_gamma', 1.0),
        )
        
        d_loss = d_losses['total'] / self.accumulate_grad
        
        # Check for NaN loss before backward
        if not torch.isfinite(d_loss):
            print(f"WARNING: NaN/Inf in D loss at step {self.global_step}, skipping batch")
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()
            return {'d_total': float('nan'), 'g_total': float('nan')}
        
        if self.use_amp:
            self.scaler.scale(d_loss).backward()
            if is_last_accum:
                self.scaler.unscale_(self.optimizer_d)
                # Check for NaN gradients
                d_grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_max_norm)
                if torch.isfinite(d_grad_norm):
                    self.scaler.step(self.optimizer_d)
                else:
                    print(f"WARNING: NaN grad in D at step {self.global_step}, skipping update")
                self.optimizer_d.zero_grad()
                # Note: scaler.update() called after G step
        else:
            d_loss.backward()
            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_max_norm)
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()
        
        real.requires_grad_(False)
        
        # ========== Train Generator ==========
        style_ref = self.style_encoder(reference, target_domain)
        
        z1 = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
        z2 = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
        style_z1 = self.mapping_network(z1, target_domain)
        style_z2 = self.mapping_network(z2, target_domain)
        
        with autocast(device_type='cuda', enabled=self.use_amp):
            # Generate fake image and get encoder features from SOURCE
            fake_ref, source_enc_features = self.generator(source, style_ref, return_features=True)
            fake_z1 = self.generator(source, style_z1)
            fake_z2 = self.generator(source, style_z2)
            
            # Cycle reconstruction
            original_domain = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            style_original = self.style_encoder(source, original_domain)
            reconstructed = self.generator(fake_ref, style_original)
            
            style_gen = self.style_encoder(fake_ref, target_domain)
            
            # Get encoder features from FAKE for NCE (compare source encoder vs fake encoder)
            # This encourages structural consistency between input and output
            _, fake_enc_features = self.generator(fake_ref, style_ref, return_features=True)
            
            # Get scores AND domain logits from fake images
            # Domain logits on fakes force G to produce domain-specific outputs
            fake_scores_g, fake_domain_logits = self.discriminator(fake_ref)
            
            g_losses = self.criterion.generator_loss(
                [s.float() for s in fake_scores_g], source, reconstructed,
                style_ref, style_gen,
                source_enc_features, fake_enc_features,  # NCE: source encoder vs fake encoder
                fake=fake_ref, target=target,
                fake_domain_logits=[logit.float() for logit in fake_domain_logits],  # Aux domain loss
                target_domain=target_domain,  # Target domain labels
            )
            
            # Style diversification loss
            div_loss = self.criterion.style_div(fake_z1, fake_z2, style_z1, style_z2)
            g_losses['ds'] = self.config['training']['losses'].get('style_diversification', 1.0) * div_loss
            g_losses['total'] = g_losses['total'] + g_losses['ds']
            
            # CRITICAL: Latent entropy regularization for mapping network
            # Forces style codes to have high entropy (diverse outputs) instead of collapsing
            # Compute entropy of style_z1 distribution to encourage spread
            lambda_entropy = self.config['training']['losses'].get('latent_entropy', 0.1)
            if lambda_entropy > 0:
                # Normalize styles to probability-like distribution
                style_probs = F.softmax(style_z1, dim=-1)
                # Entropy: - sum(p * log(p)), higher = more diverse
                entropy = -(style_probs * torch.log(style_probs + 1e-8)).sum(dim=-1).mean()
                # We want to MAXIMIZE entropy, so negate it (minimize negative entropy)
                entropy_loss = -lambda_entropy * entropy
                g_losses['entropy'] = entropy_loss
                g_losses['total'] = g_losses['total'] + entropy_loss
            
            g_loss = g_losses['total'] / self.accumulate_grad
        
        # Check for NaN loss before backward
        if not torch.isfinite(g_loss):
            print(f"WARNING: NaN/Inf in G loss at step {self.global_step}, skipping batch")
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()
            return {'d_total': float('nan'), 'g_total': float('nan')}
        
        if self.use_amp:
            self.scaler.scale(g_loss).backward()
            if is_last_accum:
                self.scaler.unscale_(self.optimizer_g)
                g_params = list(self.generator.parameters()) + list(self.style_encoder.parameters()) + list(self.mapping_network.parameters())
                g_grad_norm = torch.nn.utils.clip_grad_norm_(g_params, self.grad_clip_max_norm)
                if torch.isfinite(g_grad_norm):
                    self.scaler.step(self.optimizer_g)
                else:
                    print(f"WARNING: NaN grad in G at step {self.global_step}, skipping update")
                self.scaler.update()  # Always update scaler
                self.optimizer_g.zero_grad()
                if self.ema_g:
                    self.ema_g.update()
        else:
            g_loss.backward()
            if is_last_accum:
                g_params = list(self.generator.parameters()) + list(self.style_encoder.parameters()) + list(self.mapping_network.parameters())
                torch.nn.utils.clip_grad_norm_(g_params, self.grad_clip_max_norm)
                self.optimizer_g.step()
                self.optimizer_g.zero_grad()
                if self.ema_g:
                    self.ema_g.update()
        
        # Collect losses for logging (with NaN protection)
        losses = {}
        for k, v in d_losses.items():
            val = v.item() if torch.isfinite(v) else 0.0
            losses[f'd_{k}'] = val
        for k, v in g_losses.items():
            val = v.item() if torch.isfinite(v) else 0.0
            losses[f'g_{k}'] = val
        
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.style_encoder.train()
        self.mapping_network.train()
        
        epoch_losses = {}
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        for i, batch in enumerate(pbar):
            accum_step = i % self.accumulate_grad
            losses = self.train_step(batch, accum_step)
            
            if accum_step == self.accumulate_grad - 1:
                self.global_step += 1
            
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            pbar.set_postfix({
                'g': losses.get('g_total', 0),
                'd': losses.get('d_total', 0),
                'lr': self.optimizer_g.param_groups[0]['lr'],
            })
            
            if self.global_step % self.log_every == 0 and accum_step == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            if self.global_step % self.sample_every == 0 and accum_step == 0:
                self.save_samples(batch, self.global_step)
            
            if self.global_step % self.save_every == 0 and accum_step == 0:
                self.save_checkpoint(f'step_{self.global_step}.pth')
        
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            train_losses = self.train_epoch(epoch)
            
            for k, v in train_losses.items():
                self.writer.add_scalar(f'epoch/{k}', v, epoch)
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(f'epoch_{epoch}.pth')
            
            print(f"Epoch {epoch} - G: {train_losses.get('g_total', 0):.4f}, "
                  f"D: {train_losses.get('d_total', 0):.4f}, "
                  f"LR: {self.scheduler_g.get_last_lr()[0]:.2e}")
        
        self.save_checkpoint('final.pth')
        print("Training complete!")
    
    @torch.no_grad()
    def save_samples(self, batch: Dict[str, torch.Tensor], step: int):
        """Save sample generations using EMA weights if available."""
        self.generator.eval()
        self.style_encoder.eval()
        
        if self.ema_g:
            self.ema_g.apply_shadow()
        
        source = batch['source'][:4].to(self.device)
        reference = batch['reference'][:4].to(self.device)
        target_domain = batch['target_domain'][:4].to(self.device)
        
        style = self.style_encoder(reference, target_domain)
        fake = self.generator(source, style)
        
        source = (source + 1) / 2
        reference = (reference + 1) / 2
        fake = (fake + 1) / 2
        
        from torchvision.utils import save_image, make_grid
        grid = make_grid(torch.cat([source, reference, fake], dim=0), nrow=4)
        
        save_image(grid, self.sample_dir / f'sample_{step}.png')
        self.writer.add_image('samples', grid, step)
        
        if self.ema_g:
            self.ema_g.restore()
        
        self.generator.train()
        self.style_encoder.train()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint with validity check."""
        # First, validate that weights are not NaN
        for name, param in self.generator.named_parameters():
            if not torch.isfinite(param).all():
                print(f"WARNING: NaN detected in generator parameter '{name}'. Skipping checkpoint save.")
                return
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'mapping_network': self.mapping_network.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'best_fid': self.best_fid,
            'config': self.config,
        }
        if self.ema_g:
            checkpoint['ema_shadow'] = self.ema_g.shadow
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        self.mapping_network.load_state_dict(checkpoint['mapping_network'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        
        if self.ema_g and 'ema_shadow' in checkpoint:
            self.ema_g.shadow = checkpoint['ema_shadow']
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_fid = checkpoint.get('best_fid', float('inf'))
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
