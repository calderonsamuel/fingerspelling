# Fingerspelling Detection and Recognition

A modern implementation of fingerspelling detection and recognition using a modular multi-task architecture, replicating the approach from "Fingerspelling Detection in American Sign Language" (Shi et al., 2021) with the ChicagoFSWild dataset.

## Architecture

- **Backbone**: Custom CNN feature extractor (modular design supports YOLO integration)
- **Multi-task Heads**: 
  - Detection (classification, regression, confidence)
  - Recognition (CTC-based letter sequence prediction)
  - Pose estimation (auxiliary spatial features)
- **Metrics**: AP@IoU, AP@Acc, MSA (Mean Sequence Accuracy)
- **Losses**: Detection (focal + regression), Recognition (CTC), Letter Error Rate (REINFORCE), Pose estimation

## Features

- **Modular Design**: Clean separation of concerns with pluggable components
- **Type-Annotated**: Full type hints for better code maintainability
- **Test-Driven**: Comprehensive unit tests and architecture validation
- **Configurable**: YAML-based configuration system
- **Real Data Validation**: Tested on actual ChicagoFSWild dataset sequences

## Project Structure

```
src/
├── data/              # Data processing and loading
│   ├── preprocess.py  # ChicagoFSWild dataset preprocessing
│   └── dataset.py     # PyTorch dataset and data loaders
├── models/            # Model architectures
│   └── multitask_model.py  # Multi-task fingerspelling model
├── training/          # Training loops and losses
│   ├── trainer.py     # Training orchestration
│   └── losses.py      # All loss functions
├── evaluation/        # Metrics and evaluation
│   └── metrics.py     # AP@IoU, AP@Acc, MSA metrics
└── utils/             # Utilities and helpers
    └── types.py       # Core data types and constants
tests/                 # Unit tests
configs/               # Configuration files
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- ~2GB free disk space for ChicagoFSWild dataset

### Installation

1. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision ultralytics opencv-python numpy pandas scikit-learn matplotlib seaborn pillow tqdm pytest black mypy typing-extensions editdistance PyYAML gdown
```

3. **Download ChicagoFSWild dataset:**
```bash
python download_dataset.py
```

**Download options:**
- `--dataset-dir`: Custom extraction directory (default: `dataset/ChicagoFSWild`)
- `--skip-download`: Skip download if file already exists
- `--clean`: Clean existing dataset directory before extraction

**Manual download (if automatic fails):**
1. Download from: https://drive.google.com/file/d/1-MUy26WStlNjSEDFHN1pkP2MqD5OApFY/view?usp=sharing
2. Save as `downloads/ChicagoFSWild.tgz`
3. Run: `python download_dataset.py --skip-download`

The download script will:
- Download ChicagoFSWild.tgz (~1.8GB) from Google Drive
- Extract the main dataset with annotations and metadata
- Extract ChicagoFSWild-Frames.tgz containing all video frames
- Verify the complete dataset structure
- Display dataset statistics

## Usage

### Architecture Validation
Test the complete pipeline with a small subset:
```bash
python test_architecture.py
```

### Training

**Quick test with subset:**
```bash
python train.py --subset-size 20 --epochs 5
```

**Full training:**
```bash
python train.py --epochs 50
```

**Custom configuration:**
```bash
python train.py --config configs/custom_config.yaml --subset-size 100 --epochs 25
```

### Inference

**Test on processed frames:**
```bash
python inference.py --frames dataset/ChicagoFSWild/ChicagoFSWild-Frames/aslized/elsie_stecker_0100 --output prediction.json
```

**Live webcam inference:**
```bash
python live_inference.py --mode webcam --camera 0
```

**Video file inference:**
```bash
python live_inference.py --mode video --video path/to/your/video.mp4
```

**Live inference options:**
- Press `'q'` to quit
- Press `'s'` to save current predictions
- Adjust `--window-size` for processing (default: 30 frames)
- Use `--camera ID` to select different camera

### Configuration

Edit `configs/train_config.yaml` to customize:
- Dataset paths and image size
- Model architecture (backbone, pose estimation)
- Training parameters (batch size, learning rate, loss weights)
- Evaluation settings

## Data Format

The system expects ChicagoFSWild dataset structure:
```
dataset/ChicagoFSWild/
├── ChicagoFSWild.csv          # Main annotations
├── ChicagoFSWild-Frames/      # Video frames
└── BBox/                      # Bounding box annotations
```

Sequences are automatically split into train/dev/test partitions and processed into multi-task format with:
- Temporal detection targets (classification, regression, confidence)
- CTC-compatible recognition targets  
- Optional pose estimation targets

## Model Details

### Architecture Components
- **Backbone**: Modular CNN with configurable depth
- **Detection Head**: Multi-scale temporal detection with focal loss
- **Recognition Head**: CTC-based sequence modeling for letter prediction
- **Pose Head**: Auxiliary spatial feature learning

### Loss Functions
1. **Detection Loss**: Focal loss (classification) + smooth L1 (regression)
2. **Recognition Loss**: CTC loss for sequence alignment
3. **Letter Error Rate**: REINFORCE-based policy gradient loss
4. **Pose Loss**: MSE for spatial feature consistency

### Metrics
- **AP@IoU**: Average Precision at IoU thresholds (0.1, 0.3, 0.5)
- **AP@Acc**: Average Precision at accuracy thresholds (0.0, 0.2, 0.4) 
- **MSA**: Mean Sequence Accuracy for recognition quality

## Development

### Testing
```bash
# Run unit tests
pytest tests/

# Test specific module
pytest tests/test_types.py -v

# Architecture validation
python test_architecture.py
```

### Code Quality
```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/
```

This project follows test-driven development practices with comprehensive validation on real data.

## Dataset Download

### Automatic Download
The easiest way to get the dataset:
```bash
python download_dataset.py
```

### Manual Download
If automatic download fails:
1. Go to: https://drive.google.com/file/d/1-MUy26WStlNjSEDFHN1pkP2MqD5OApFY/view?usp=sharing
2. Download `ChicagoFSWild.tgz` to `downloads/` folder
3. Run: `python download_dataset.py --skip-download`

### Troubleshooting Download Issues

**"gdown" not found:**
```bash
pip install gdown
```

**Google Drive download limit:**
- Try again later (Google has daily download limits)
- Use manual download method above

**Extraction errors:**
```bash
# Clean and retry
python download_dataset.py --clean
```

**Verification of dataset:**
After download, you should have:
```
dataset/ChicagoFSWild/
├── ChicagoFSWild.csv          # 7,306 sequences
├── ChicagoFSWild-Frames/      # Video frames (~16 subdirs)
├── BBox/                      # Bounding box annotations
├── README                     # Dataset documentation
└── *.csv                      # Various metadata files
```

Check the dataset with:
```bash
python -c "
import pandas as pd
df = pd.read_csv('dataset/ChicagoFSWild/ChicagoFSWild.csv')
print(f'Dataset loaded: {len(df)} sequences')
print(f'Partitions: {df[\"partition\"].value_counts().to_dict()}')
"
```
