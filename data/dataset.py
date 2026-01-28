"""
IHC4BC Dataset Loader

Handles loading of H&E and IHC image pairs for multi-domain virtual staining.
Adapted for the actual IHC4BC dataset structure.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class IHC4BCDataset(Dataset):
    """
    Dataset for IHC4BC multi-domain virtual staining.
    
    Actual structure:
        root/
        ├── Images/
        │   ├── HandE/           # H&E stained images
        │   │   ├── ER/
        │   │   │   ├── Patient_X/
        │   │   │   │   └── Subregion_Y/
        │   │   │   │       └── *.jpg
        │   │   ├── Her2/
        │   │   ├── Ki67/
        │   │   └── PR/
        │   └── IHC/             # Corresponding IHC images
        │       ├── ER/
        │       ├── Her2/
        │       ├── Ki67/
        │       └── PR/
        └── Labels/
    """
    
    # Domain names as they appear in folder structure
    DOMAINS = ["ER", "PR", "Ki67", "Her2"]
    DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}
    IDX_TO_DOMAIN = {i: d for i, d in enumerate(DOMAINS)}
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        domains: Optional[List[str]] = None,
        paired: bool = True,
        paired_only: bool = False,
        paired_reference: bool = False,
        image_size: int = 256,
    ):
        """
        Args:
            root_dir: Root directory of dataset (IHC4BC_Compressed folder)
            split: One of 'train', 'val', 'test'
            transform: Transforms for source (H&E) images
            target_transform: Transforms for target (IHC) images
            domains: List of domains to include (default: all)
            paired: If True, return paired H&E-IHC samples
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.domains = domains or self.DOMAINS
        self.paired = paired
        self.paired_only = paired_only
        self.paired_reference = paired_reference
        self.image_size = image_size
        
        # Find the Images directory
        self.images_dir = self.root_dir / "Images"
        if not self.images_dir.exists():
            # Maybe root already points to Images
            if (self.root_dir / "HandE").exists():
                self.images_dir = self.root_dir
            else:
                raise FileNotFoundError(f"Images directory not found in {root_dir}")
        
        self.he_dir = self.images_dir / "HandE"
        self.ihc_dir = self.images_dir / "IHC"
        
        if not self.he_dir.exists():
            raise FileNotFoundError(f"HandE directory not found: {self.he_dir}")

        if not self.ihc_dir.exists():
            ihc_candidates = [p for p in self.images_dir.iterdir() if p.is_dir() and p.name.lower() == "ihc"]
            if ihc_candidates:
                self.ihc_dir = ihc_candidates[0]
            else:
                raise FileNotFoundError(
                    f"IHC directory not found under {self.images_dir}. "
                    f"Expected 'IHC' alongside 'HandE'. Check DATASET_PATH."
                )
        
        # Load samples
        self.samples = self._load_samples()
        
        # Create domain-specific image lists for reference sampling
        self.domain_images = self._build_domain_index()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        for domain in self.domains:
            count = len(self.domain_images.get(domain, []))
            print(f"  {domain}: {count} reference images")
    
    def _load_samples(self) -> List[Dict]:
        """Load all paired image samples."""
        samples = []

        he_domain_map = {p.name.lower(): p.name for p in self.he_dir.iterdir() if p.is_dir()}
        ihc_domain_map = {p.name.lower(): p.name for p in self.ihc_dir.iterdir() if p.is_dir()}
        
        for domain in self.domains:
            he_domain_name = he_domain_map.get(domain.lower(), domain)
            ihc_domain_name = ihc_domain_map.get(domain.lower(), domain)

            he_domain_dir = self.he_dir / he_domain_name
            ihc_domain_dir = self.ihc_dir / ihc_domain_name
            
            if not he_domain_dir.exists():
                print(f"Warning: HandE/{domain} not found, skipping")
                continue
            
            # Collect all H&E images for this domain
            he_images = []
            for patient_dir in he_domain_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                for subregion_dir in patient_dir.iterdir():
                    if not subregion_dir.is_dir():
                        continue
                    for img_path in subregion_dir.glob("*.jpg"):
                        he_images.append((img_path, domain))
            
            # Shuffle deterministically for reproducible splits
            random.seed(42)
            random.shuffle(he_images)
            
            # Split dataset
            n_total = len(he_images)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            
            if self.split == "train":
                split_images = he_images[:n_train]
            elif self.split == "val":
                split_images = he_images[n_train:n_train + n_val]
            else:  # test
                split_images = he_images[n_train + n_val:]
            
            # Build samples with paired IHC images
            for he_path, domain in split_images:
                # Construct corresponding IHC path
                rel_path = he_path.relative_to(he_domain_dir)
                ihc_path = ihc_domain_dir / rel_path
                
                sample = {
                    "he_path": he_path,
                    "domain": domain,
                    "domain_idx": self.DOMAIN_TO_IDX[domain],
                }
                
                if ihc_path.exists():
                    sample["ihc_path"] = ihc_path
                    samples.append(sample)
                elif not self.paired_only:
                    samples.append(sample)
        
        # Shuffle samples
        random.seed(42 + hash(self.split))
        random.shuffle(samples)
        
        return samples
    
    def _build_domain_index(self) -> Dict[str, List[Path]]:
        """Build index of IHC images per domain for reference sampling."""
        domain_images = {d: [] for d in self.domains}

        ihc_domain_map = {p.name.lower(): p.name for p in self.ihc_dir.iterdir() if p.is_dir()}
        
        for domain in self.domains:
            ihc_domain_name = ihc_domain_map.get(domain.lower(), domain)
            ihc_domain_dir = self.ihc_dir / ihc_domain_name
            if not ihc_domain_dir.exists():
                continue
            
            # Collect all IHC images
            for img_path in ihc_domain_dir.rglob("*.jpg"):
                domain_images[domain].append(img_path)
        
        return domain_images
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load and resize image."""
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - 'source': H&E image tensor
                - 'target': IHC image tensor (if paired and available)
                - 'target_domain': Domain index
                - 'reference': Reference image from target domain (for style)
        """
        sample = self.samples[idx]
        
        # Load H&E (source) image
        source = self._load_image(sample["he_path"])
        
        domain = sample["domain"]
        domain_idx = sample["domain_idx"]
        
        # Load target IHC image if available
        target = None
        if self.paired and "ihc_path" in sample:
            target = self._load_image(sample["ihc_path"])
        
        # Reference image for style
        if self.paired_reference and target is not None:
            reference = target
        else:
            ref_images = self.domain_images.get(domain, [])
            if ref_images:
                ref_path = random.choice(ref_images)
                reference = self._load_image(ref_path)
            else:
                reference = target if target is not None else source
        
        # Apply transforms
        if self.transform:
            source = self.transform(source)
        else:
            source = self._to_tensor(source)
        
        if target is not None:
            if self.target_transform:
                target = self.target_transform(target)
            else:
                target = self._to_tensor(target)
        
        if self.transform:
            reference = self.transform(reference)
        else:
            reference = self._to_tensor(reference)
        
        result = {
            "source": source,
            "target_domain": torch.tensor(domain_idx, dtype=torch.long),
            "reference": reference,
        }
        
        if target is not None:
            result["target"] = target
        
        return result
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)
    
    @staticmethod
    def get_domain_name(idx: int) -> str:
        """Get domain name from index."""
        return IHC4BCDataset.IDX_TO_DOMAIN.get(idx, "Unknown")
    
    @staticmethod
    def get_domain_idx(name: str) -> int:
        """Get domain index from name."""
        return IHC4BCDataset.DOMAIN_TO_IDX.get(name, -1)


def get_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    transform=None,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for IHC4BC dataset."""
    dataset = IHC4BCDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
