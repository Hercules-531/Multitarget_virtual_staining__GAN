"""
Data Transforms for Histopathology Images

Includes stain normalization, augmentation, and preprocessing.
"""

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256):
    """Training transforms with augmentation (for unpaired data only)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=1.0
            ),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), mean=0.0, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=0.2
        ),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def get_paired_train_transforms(image_size: int = 256):
    """
    Training transforms for PAIRED data.
    
    CRITICAL: For paired training with L1 loss, source and target must have
    IDENTICAL spatial transformations. We only use geometric augmentations
    that can be applied consistently to both images.
    
    NO color jitter - we want to learn the color mapping from H&E to IHC!
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Only geometric augmentations (applied identically to source & target)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Normalize to [-1, 1]
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={'target': 'image', 'reference': 'image'})


def get_val_transforms(image_size: int = 256):
    """Validation/test transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


class AlbumentationsWrapper:
    """Wrapper to make albumentations work with PIL Images."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    tensor = denormalize(tensor)
    tensor = tensor.clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


class StainNormalizer:
    """
    Stain normalization using Macenko method.
    
    Normalizes H&E images to a reference stain appearance.
    """
    
    def __init__(self, reference_image: np.ndarray = None):
        self.reference_image = reference_image
        self.stain_matrix_ref = None
        self.max_conc_ref = None
        
        if reference_image is not None:
            self._fit(reference_image)
    
    def _fit(self, image: np.ndarray):
        """Fit normalizer to reference image."""
        # Convert to OD space
        image = image.astype(np.float32) + 1
        od = -np.log(image / 255.0)
        
        # Get stain matrix using SVD
        od_flat = od.reshape(-1, 3)
        od_flat = od_flat[np.all(od_flat > 0.15, axis=1)]
        
        if len(od_flat) > 0:
            _, _, vh = np.linalg.svd(od_flat, full_matrices=False)
            self.stain_matrix_ref = vh[:2, :]
            
            # Get concentration
            conc = np.linalg.lstsq(self.stain_matrix_ref.T, od_flat.T, rcond=None)[0]
            self.max_conc_ref = np.percentile(conc, 99, axis=1)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to reference stain appearance."""
        if self.stain_matrix_ref is None:
            return image
        
        # Simple color normalization fallback
        # Full Macenko requires more complex implementation
        img_lab = self._rgb_to_lab(image)
        ref_lab = self._rgb_to_lab(self.reference_image) if self.reference_image is not None else img_lab
        
        # Match mean and std in LAB space
        img_mean = img_lab.mean(axis=(0, 1))
        img_std = img_lab.std(axis=(0, 1)) + 1e-6
        ref_mean = ref_lab.mean(axis=(0, 1))
        ref_std = ref_lab.std(axis=(0, 1)) + 1e-6
        
        normalized = (img_lab - img_mean) / img_std * ref_std + ref_mean
        return self._lab_to_rgb(normalized)
    
    def _rgb_to_lab(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        from skimage import color
        return color.rgb2lab(img / 255.0)
    
    def _lab_to_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert LAB to RGB color space."""
        from skimage import color
        rgb = color.lab2rgb(img)
        return (rgb * 255).astype(np.uint8)
