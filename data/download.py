"""
Kaggle Dataset Download Script for IHC4BC

Before running, ensure you have:
1. Kaggle API key configured (~/.kaggle/kaggle.json)
2. Accepted the dataset terms on Kaggle

Usage:
    python data/download.py --output ./data/ihc4bc
"""

import os
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm


def download_dataset(output_dir: str = "./data/ihc4bc"):
    """Download IHC4BC dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Please install kaggle: pip install kaggle")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "akbarnejad1991/ihc4bc-compressed"
    print(f"Downloading {dataset_name}...")
    print("Note: This is a 37GB dataset. Download may take a while.")
    
    api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=False,
        quiet=False
    )
    
    # Find and extract zip files
    zip_files = list(output_path.glob("*.zip"))
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in tqdm(zf.namelist(), desc="Extracting"):
                zf.extract(member, output_path)
        
        # Remove zip after extraction
        zip_file.unlink()
    
    print(f"Dataset downloaded and extracted to {output_path}")
    return True


def organize_dataset(data_dir: str = "./data/ihc4bc"):
    """
    Organize the dataset into the expected structure:
    
    ihc4bc/
    ├── HE/           # H&E stained images
    ├── ER/           # ER IHC images
    ├── PR/           # PR IHC images
    ├── Ki67/         # Ki67 IHC images
    ├── HER2/         # HER2 IHC images
    └── metadata/     # CSV files with annotations
    """
    data_path = Path(data_dir)
    
    # Create domain directories
    domains = ["HE", "ER", "PR", "Ki67", "HER2"]
    for domain in domains:
        (data_path / domain).mkdir(exist_ok=True)
    (data_path / "metadata").mkdir(exist_ok=True)
    
    # The IHC4BC dataset structure may vary
    # This function handles the organization based on actual structure
    print("Organizing dataset...")
    
    # Move CSV files to metadata
    for csv_file in data_path.rglob("*.csv"):
        if csv_file.parent != data_path / "metadata":
            target = data_path / "metadata" / csv_file.name
            if not target.exists():
                csv_file.rename(target)
    
    print("Dataset organization complete.")
    print("\nDataset structure:")
    for domain in domains:
        count = len(list((data_path / domain).glob("*")))
        print(f"  {domain}: {count} images")


def verify_dataset(data_dir: str = "./data/ihc4bc"):
    """Verify dataset integrity."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Dataset directory not found: {data_path}")
        return False
    
    domains = ["HE", "ER", "PR", "Ki67", "HER2"]
    all_good = True
    
    for domain in domains:
        domain_path = data_path / domain
        if domain_path.exists():
            count = len(list(domain_path.glob("*.png"))) + len(list(domain_path.glob("*.jpg")))
            print(f"{domain}: {count} images")
            if count == 0:
                print(f"  Warning: No images found in {domain}")
                all_good = False
        else:
            print(f"{domain}: Directory not found")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IHC4BC dataset")
    parser.add_argument("--output", type=str, default="./data/ihc4bc",
                        help="Output directory for dataset")
    parser.add_argument("--organize", action="store_true",
                        help="Only organize existing dataset")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify dataset")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output)
    elif args.organize:
        organize_dataset(args.output)
    else:
        if download_dataset(args.output):
            organize_dataset(args.output)
            verify_dataset(args.output)
