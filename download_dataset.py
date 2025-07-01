"""
Script to download and extract the ChicagoFSWild dataset.
Downloads from Google Drive and extracts all necessary files.
"""

import os
import sys
import subprocess
import tarfile
import zipfile
from pathlib import Path
import urllib.request
import shutil
import argparse


def install_gdown():
    """Install gdown for Google Drive downloads."""
    print("Installing gdown for Google Drive downloads...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("✅ gdown installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install gdown")
        return False


def download_from_google_drive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive using gdown."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"Downloading from Google Drive to {output_path}...")
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            print(f"✅ Downloaded successfully: {output_path}")
            return True
        else:
            print(f"❌ Download failed: {output_path}")
            return False
            
    except ImportError:
        print("❌ gdown not available, trying alternative method...")
        return download_with_urllib(file_id, output_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_with_urllib(file_id: str, output_path: str) -> bool:
    """Alternative download method using urllib."""
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print(f"Downloading with urllib to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        
        if os.path.exists(output_path):
            print(f"✅ Downloaded successfully: {output_path}")
            return True
        else:
            print(f"❌ Download failed: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Download with urllib failed: {e}")
        return False


def extract_tar_gz(tar_path: str, extract_to: str) -> bool:
    """Extract .tar.gz file."""
    try:
        print(f"Extracting {tar_path} to {extract_to}...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✅ Extracted successfully: {tar_path}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def extract_tar(tar_path: str, extract_to: str) -> bool:
    """Extract .tar file."""
    try:
        print(f"Extracting {tar_path} to {extract_to}...")
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✅ Extracted successfully: {tar_path}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def check_extraction(dataset_dir: Path) -> bool:
    """Check if extraction was successful."""
    required_files = [
        "ChicagoFSWild.csv",
        "annotation_instructions.txt",
        "batch.csv",
        "individual.csv",
        "unavailable.csv",
        "getChicagoFSWild.py",
        "HandAnnotation.csv",
        "README"
    ]
    
    required_dirs = [
        "BBox",
        "ChicagoFSWild-Frames"
    ]
    
    print("Checking extracted files...")
    
    missing_files = []
    for file in required_files:
        file_path = dataset_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✅ Found: {dir_name}/")
    
    if missing_files or missing_dirs:
        print("\n❌ Missing files/directories:")
        for item in missing_files + missing_dirs:
            print(f"  - {item}")
        return False
    
    # Check if ChicagoFSWild-Frames has content
    frames_dir = dataset_dir / "ChicagoFSWild-Frames"
    if frames_dir.exists():
        subdirs = list(frames_dir.glob("*"))
        if len(subdirs) > 0:
            print(f"✅ ChicagoFSWild-Frames contains {len(subdirs)} subdirectories")
        else:
            print("⚠️  ChicagoFSWild-Frames is empty")
    
    print("\n✅ Dataset extraction verification complete!")
    return True


def get_dataset_info(dataset_dir: Path):
    """Get information about the dataset."""
    csv_path = dataset_dir / "ChicagoFSWild.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            print(f"\n📊 Dataset Information:")
            print(f"  Total sequences: {len(df)}")
            print(f"  Partitions: {df['partition'].value_counts().to_dict()}")
            print(f"  Unique signers: {df['signer'].nunique()}")
            print(f"  Average sequence length: {df['number_of_frames'].mean():.1f} frames")
            
        except ImportError:
            print("\n📊 Install pandas to see detailed dataset statistics")
        except Exception as e:
            print(f"\n📊 Could not read dataset info: {e}")


def main():
    """Main function to download and extract dataset."""
    parser = argparse.ArgumentParser(description='Download and extract ChicagoFSWild dataset')
    parser.add_argument('--dataset-dir', type=str, default='dataset/ChicagoFSWild',
                       help='Directory to extract dataset (default: dataset/ChicagoFSWild)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download if file already exists')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing dataset directory before extraction')
    
    args = parser.parse_args()
    
    print("🎯 ChicagoFSWild Dataset Setup")
    print("=" * 40)
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    downloads_dir = Path("downloads")
    tar_file = downloads_dir / "ChicagoFSWild.tgz"
    
    # Create directories
    downloads_dir.mkdir(exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean if requested
    if args.clean and dataset_dir.exists():
        print(f"🧹 Cleaning existing dataset directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    # Google Drive file ID from the shared link
    file_id = "1-MUy26WStlNjSEDFHN1pkP2MqD5OApFY"
    
    # Download dataset
    if not args.skip_download or not tar_file.exists():
        print(f"📥 Downloading ChicagoFSWild.tgz...")
        
        # Try to install gdown first
        try:
            import gdown
        except ImportError:
            if not install_gdown():
                print("❌ Could not install gdown, please install manually:")
                print("   pip install gdown")
                return False
        
        if not download_from_google_drive(file_id, str(tar_file)):
            print("❌ Download failed!")
            print("\n🔗 Manual download instructions:")
            print("1. Go to: https://drive.google.com/file/d/1-MUy26WStlNjSEDFHN1pkP2MqD5OApFY/view?usp=sharing")
            print(f"2. Download ChicagoFSWild.tgz to: {tar_file}")
            print("3. Re-run this script with --skip-download")
            return False
    else:
        print(f"✅ Using existing file: {tar_file}")
    
    # Verify download
    if not tar_file.exists():
        print(f"❌ File not found: {tar_file}")
        return False
    
    file_size = tar_file.stat().st_size / (1024 * 1024 * 1024)  # GB
    print(f"📁 File size: {file_size:.2f} GB")
    
    # Extract main dataset
    print(f"📦 Extracting dataset to {dataset_dir}...")
    if not extract_tar_gz(str(tar_file), str(dataset_dir.parent)):
        return False
    
    # Check if ChicagoFSWild-Frames.tgz exists and extract it
    frames_tar = dataset_dir / "ChicagoFSWild-Frames.tgz"
    if frames_tar.exists():
        print(f"📦 Found ChicagoFSWild-Frames.tgz, extracting...")
        # Create the ChicagoFSWild-Frames directory
        frames_extract_dir = dataset_dir / "ChicagoFSWild-Frames"
        frames_extract_dir.mkdir(exist_ok=True)
        
        if not extract_tar_gz(str(frames_tar), str(frames_extract_dir)):
            print("⚠️  Failed to extract frames, but continuing...")
        else:
            # Remove the .tgz file after extraction to save space
            frames_tar.unlink()
            print("🗑️  Removed ChicagoFSWild-Frames.tgz after extraction")
    else:
        print("⚠️  ChicagoFSWild-Frames.tgz not found in extracted data")
    
    # Verify extraction
    if not check_extraction(dataset_dir):
        print("❌ Dataset extraction verification failed!")
        return False
    
    # Get dataset information
    get_dataset_info(dataset_dir)
    
    print(f"\n🎉 Dataset setup complete!")
    print(f"📂 Dataset location: {dataset_dir.absolute()}")
    print(f"\n🚀 Next steps:")
    print(f"  1. Test the architecture: python test_architecture.py")
    print(f"  2. Train a model: python train.py --subset-size 20 --epochs 5")
    print(f"  3. Run inference: python inference.py --frames {dataset_dir}/ChicagoFSWild-Frames/aslized/elsie_stecker_0001")
    
    # Optional cleanup
    response = input(f"\n🗑️  Remove downloaded .tgz file to save space? (y/n): ")
    if response.lower() in ['y', 'yes']:
        tar_file.unlink()
        print(f"✅ Removed {tar_file}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
