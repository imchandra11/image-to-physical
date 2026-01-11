# EasyOCRTraining: OCR Text Recognition Training Framework

A PyTorch Lightning-based training framework for OCR (Optical Character Recognition) text recognition models. This project implements a CRNN (CNN-RNN) architecture with CTC or Attention-based decoding for recognizing text in images.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Format](#data-format)
- [EasyOCR Model](#easyocr-model)
- [Module Documentation](#module-documentation)
- [Key Concepts](#key-concepts)
- [Workflow](#workflow)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

This project implements an OCR text recognition system using a **CRNN (CNN-RNN)** architecture. The system consists of four stages: **Transformation** (optional TPS), **Feature Extraction** (VGG/ResNet/RCNN), **Sequence Modeling** (BiLSTM), and **Prediction** (CTC or Attention).

The system is built on **PyTorch Lightning**, which provides a clean abstraction for training, validation, testing, and prediction workflows. It uses **JSONArgParse** (via Lightning CLI) for flexible configuration management through YAML files.

### Key Technologies

- **PyTorch Lightning 2.5.1**: Training framework
- **CRNN Architecture**: CNN-RNN for text recognition
- **CTC Decoding**: Connectionist Temporal Classification for sequence labeling
- **Attention Mechanism**: Attention-based sequence-to-sequence decoding
- **OpenCV/PIL**: Image processing
- **JSONArgParse**: Configuration management
- **jiwer**: Word and Character Error Rate metrics

### What is OCR Text Recognition?

OCR (Optical Character Recognition) text recognition is the process of:
1. **Transformation** (optional): Rectify distorted text images using TPS (Thin-Plate Spline)
2. **Feature Extraction**: Extract visual features from text images using CNN (VGG/ResNet/RCNN)
3. **Sequence Modeling**: Model sequential dependencies using RNN (BiLSTM)
4. **Prediction**: Predict character sequences using CTC or Attention decoding

## Project Structure

```
EasyOCRTraining/
│
├── config/                      # Configuration files
│   └── easyOCR.yaml            # Main configuration file
│
├── utilities/                   # Model component utilities
│   ├── feature_extraction.py   # VGG, ResNet, RCNN feature extractors
│   ├── sequence_modeling.py    # BidirectionalLSTM
│   ├── prediction.py           # Attention module
│   ├── transformation.py       # TPS Spatial Transformer Network
│   └── labelconverter.py       # CTC and Attention label converters
│
├── cli.py                       # Custom Lightning CLI (EasyOCRLightningCLI)
├── datamodule.py                # Lightning DataModule for data handling
├── dataset.py                   # OCRDataset, AlignCollate
├── main.py                      # Main entry point for CLI commands
├── main_fittest.py              # Entry point for fit+test workflow
├── model.py                     # EasyOCRNet - main model architecture
├── module.py                    # EasyOCRLitModule - Lightning Module for training
└── README.md                    # This file

utils/                           # Shared utility modules
└── easyocrmetrics.py           # WER/CER evaluation metrics

EasyOCRTrainer.ipynb            # Jupyter notebook for interactive use
```

## Architecture

### High-Level Architecture

The system follows a **modular, plugin-based architecture**:

1. **Core Framework (EasyOCRTraining/)**: Provides reusable components for OCR training
   - CRNN model implementation (EasyOCRNet)
   - Lightning module for training logic
   - Data module for data loading
   - CLI for command-line interface

2. **Utilities (utilities/)**: Model component implementations
   - Feature extractors (VGG, ResNet, RCNN)
   - Sequence modeling (BiLSTM)
   - Prediction modules (CTC, Attention)
   - Transformation (TPS)
   - Label converters (CTC, Attention)

3. **Shared Utilities (utils/)**: Shared helper functions
   - Metrics computation (WER, CER)

### Design Patterns

- **Factory Pattern**: Model component creation via configuration
- **Template Method Pattern**: Base classes (LightningModule, LightningDataModule) define structure
- **Strategy Pattern**: Different prediction strategies (CTC vs Attention)
- **Plugin Architecture**: Feature extractors and prediction methods are pluggable

## Features

### Core Features

1. **CRNN Model Implementation**
   - Four-stage architecture: Transformation → Feature Extraction → Sequence Modeling → Prediction
   - Multiple feature extractors: VGG, ResNet, RCNN
   - Bidirectional LSTM for sequence modeling
   - CTC or Attention-based decoding

2. **Flexible Architecture**
   - Optional TPS transformation for text rectification
   - Multiple feature extractor backbones
   - Configurable sequence modeling
   - CTC or Attention prediction

3. **Complete Training Pipeline**
   - Training with validation
   - Model checkpointing (best and last)
   - Comprehensive evaluation metrics (WER, CER, Accuracy)
   - Prediction/inference mode

4. **Image Processing**
   - Aspect-ratio preserving resize with padding
   - Grayscale or RGB input
   - Configurable image dimensions

5. **Flexible Configuration**
   - YAML-based configuration
   - Local config overrides
   - Command-line argument support
   - Configurable model, data, and training parameters

### Training Features

- **Automatic mixed precision** (16-mixed precision for GPU)
- **Checkpoint management**: Best and last checkpoints
- **TensorBoard logging**: Metrics and visualizations
- **Validation during training**: Per-epoch validation
- **Freezing options**: Freeze feature extraction or sequence modeling layers

### Evaluation Features

- **WER/CER metrics**: Word Error Rate and Character Error Rate
- **Accuracy computation**: Exact string matching
- **Prediction export**: CSV files with ground truth and predictions
- **Metrics logging**: WER and CER logged to TensorBoard

## Requirements

### System Requirements

- **Python**: 3.12.10
- **CUDA**: 12.6.3 (for GPU support)
- **Operating System**: Windows/Linux/MacOS

### Python Dependencies

See `requirements.txt` in project root. Key dependencies:

- `torch==2.6.0+cu126`
- `torchvision==0.21.0+cu126`
- `lightning==2.5.1.post0` (PyTorch Lightning)
- `Pillow==11.0.0`
- `pandas==2.2.3`
- `jsonargparse==4.40.0`
- `tensorboard==2.19.0`
- `jiwer` (for WER/CER metrics)

## Installation

### Step 1: Install PyTorch with CUDA Support

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### Step 2: Install Lightning

```bash
pip install lightning
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

Or install key packages individually:

```bash
pip install Pillow pandas jsonargparse tensorboard jiwer
```

## Configuration

### Configuration Files

The project uses YAML configuration files:

1. **Main Config** (`EasyOCRTraining/config/easyOCR.yaml`): Main configuration

### Key Configuration Sections

#### Trainer Configuration

```yaml
trainer:
  max_epochs: 2
  accelerator: gpu
  devices: 1
  precision: 16-mixed  # Mixed precision for GPU
  callbacks:
    - ModelCheckpoint for best model
    - ModelCheckpoint for last model
  logger:
    TensorBoardLogger configuration
```

#### Model Configuration

```yaml
model:
  class_path: EasyOCRTraining.module.EasyOCRLitModule
  init_args:
    Prediction: "CTC"  # or "Attn"
    batch_max_length: 40
    character: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&'()*+,-./:;<=>?@[]^_`{|}~"
    pretrained_model: "path/to/pretrained.pth"  # Optional
    decode: "greedy"  # or "beamsearch" (for CTC)
    freeze_FeatureExtraction: false
    freeze_SequenceModeling: false
    model:
      class_path: EasyOCRTraining.model.EasyOCRNet
      init_args:
        Transformation: "None"  # or "TPS"
        FeatureExtraction: "VGG"  # or "ResNet", "RCNN"
        SequenceModeling: "BiLSTM"  # or "None"
        Prediction: "CTC"  # or "Attn"
        input_channel: 1
        output_channel: 512
        hidden_size: 256
        num_fiducial: 20  # for TPS
        imgH: 32
        imgW: 100
```

#### Data Configuration

```yaml
data:
  class_path: EasyOCRTraining.datamodule.EasyOCRDataModule
  init_args:
    data_dir: E:/EasyOCRData
    batch_size: 1
    num_workers: 4
    imgH: 32
    imgW: 100
    PAD: true  # Keep aspect ratio with padding
    contrast_adjust: 0.0
```

#### Optimizer

```yaml
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
```

## Usage

### Data Directory Structure

Your data should be organized as follows:

```
data_dir/
├── Training/
│   ├── images/          # Training images
│   └── labels.csv       # CSV with 'filename' and 'words' columns
├── Validation/
│   ├── images/
│   └── labels.csv
└── Testing/
    ├── images/
    └── labels.csv
```

**Note**: The CSV file should contain at least two columns:
- `filename`: Name of the image file
- `words`: Ground truth text label

**Example CSV (`labels.csv`)**:
```csv
filename,words
img001.png,hello world
img002.png,deep learning
img003.png,optical character recognition
```

### Training

#### Using CLI (Recommended)

```bash
python EasyOCRTraining/main.py fit \
  --config EasyOCRTraining/config/easyOCR.yaml
```

#### Using main_fittest.py (Fit + Test)

```bash
python EasyOCRTraining/main_fittest.py \
  --config EasyOCRTraining/config/easyOCR.yaml
```

#### Using Jupyter Notebook

See `EasyOCRTrainer.ipynb` for interactive usage:

```python
from EasyOCRTraining.module import EasyOCRLitModule
from EasyOCRTraining.datamodule import EasyOCRDataModule

%run EasyOCRTraining/main_fittest.py \
  --config EasyOCRTraining/config/easyOCR.yaml
```

### Resuming Training

To resume from a checkpoint:

```bash
python EasyOCRTraining/main.py fit \
  --config EasyOCRTraining/config/easyOCR.yaml \
  --fit.ckpt_path path/to/checkpoint.ckpt
```

Or add to config:

```yaml
fit:
  ckpt_path: path/to/checkpoint.ckpt
```

### Testing

```bash
python EasyOCRTraining/main.py test \
  --config EasyOCRTraining/config/easyOCR.yaml \
  --test.ckpt_path best  # or 'last' or path to checkpoint
```

**Note**: Test results are saved to `prediction/output.csv` in the log directory.

### Prediction/Inference

```bash
python EasyOCRTraining/main.py predict \
  --config EasyOCRTraining/config/easyOCR.yaml \
  --data.data_dir /path/to/test/images \
  --test.ckpt_path path/to/checkpoint.ckpt
```

**Note**: Prediction results are saved to `prediction/predict_epoch{N}.csv` in the log directory.

## Data Format

### Input Images

- **Supported formats**: PNG, JPG, JPEG (via PIL)
- **Processing pipeline**: 
  1. Load image (PIL)
  2. Convert to grayscale or RGB
  3. Resize with aspect ratio preservation (if `PAD: true`)
  4. Pad to fixed dimensions
  5. Convert to tensor

### Ground Truth Format

CSV files (`labels.csv`) with format:
- **Required columns**: `filename`, `words`
- Each row: image filename and corresponding text label
- Text labels can contain any characters in the character set

### Output Format

#### Training/Validation

- Loss values logged to TensorBoard
- Validation accuracy logged
- WER and CER logged at end of validation epoch
- Checkpoints saved automatically

#### Testing

- **CSV file**: `prediction/output.csv` with columns:
  - `ground_truth`: Ground truth text
  - `predicted_text`: Predicted text
- **Metrics**: WER and CER computed and logged

#### Prediction

- **CSV file**: `prediction/predict_epoch{N}.csv` with same format as testing

## EasyOCR Model

### Model Architecture

The EasyOCR model (`EasyOCRNet`) consists of four stages:

1. **Transformation Stage** (Optional):
   - TPS (Thin-Plate Spline) Spatial Transformer Network
   - Rectifies distorted text images
   - Configurable number of fiducial points

2. **Feature Extraction Stage**:
   - **VGG_FeatureExtractor**: VGG-like CNN (default)
   - **ResNet_FeatureExtractor**: ResNet-based CNN
   - **RCNN_FeatureExtractor**: Gated RCNN (GRCL)
   - Extracts visual features from images
   - Output: Feature maps (B, C, H, W)

3. **Sequence Modeling Stage**:
   - **BidirectionalLSTM**: Bidirectional LSTM (default)
   - Models sequential dependencies
   - Converts spatial features to sequence features
   - Output: Contextual features (B, T, hidden_size)

4. **Prediction Stage**:
   - **CTC**: Connectionist Temporal Classification (default)
     - Linear layer mapping to character classes
     - Handles variable-length sequences
   - **Attention**: Attention-based sequence-to-sequence
     - Attention mechanism for alignment
     - Teacher forcing for training

### Model Features

- **Configurable Components**: Each stage can be configured independently
- **Multiple Backbones**: Support for VGG, ResNet, and RCNN feature extractors
- **Flexible Decoding**: CTC or Attention-based prediction
- **Pretrained Support**: Load pretrained model weights
- **Freezing Options**: Freeze feature extraction or sequence modeling layers

### Architecture Options

#### Transformation
- `"None"`: No transformation
- `"TPS"`: Thin-Plate Spline rectification

#### Feature Extraction
- `"VGG"`: VGG-like CNN (default, fast)
- `"ResNet"`: ResNet-based CNN (deeper, more parameters)
- `"RCNN"`: Gated RCNN (GRCL layers)

#### Sequence Modeling
- `"BiLSTM"`: Bidirectional LSTM (default)
- `"None"`: No sequence modeling (direct prediction)

#### Prediction
- `"CTC"`: CTC decoding (default, handles variable length)
- `"Attn"`: Attention-based decoding (sequence-to-sequence)

## Module Documentation

### EasyOCRTraining Module

#### `cli.py` - EasyOCRLightningCLI

Custom CLI class extending LightningCLI with:
- Checkpoint path arguments for fit and test

#### `datamodule.py` - EasyOCRDataModule

PyTorch Lightning DataModule that:
- Manages train/val/test/predict datasets
- Handles data loading with proper transforms
- Uses AlignCollate for batch processing with padding
- Provides data loaders with aspect-ratio preserving resize

**Key Methods**:
- `setup(stage)`: Prepares datasets for different stages
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `predict_dataloader()`
- `transfer_batch_to_device()`: Custom device transfer

#### `dataset.py` - OCRDataset, AlignCollate

**OCRDataset**:
- Loads images and labels from CSV files
- Returns PIL images and text labels
- Supports grayscale or RGB images

**AlignCollate**:
- Collate function for variable-size images
- Resizes images while preserving aspect ratio
- Pads to fixed dimensions (imgW x imgH)
- Returns batched tensors and label lists

#### `module.py` - EasyOCRLitModule

PyTorch Lightning Module for OCR training:
- Wraps EasyOCRNet model
- Handles training, validation, testing, and prediction
- Supports both CTC and Attention prediction
- Computes loss (CTCLoss or CrossEntropyLoss)
- Performs decoding (greedy or beamsearch for CTC)
- Computes WER/CER metrics
- Saves predictions to CSV

**Key Methods**:
- `training_step()`: Forward pass and loss computation
- `validation_step()`: Validation with loss, accuracy, and metrics
- `on_validation_epoch_end()`: Compute and log WER/CER
- `test_step()`: Inference with predictions
- `predict_step()`: Inference without metrics
- `_runInference()`: Core inference logic
- `_preparePrediction()`: Setup prediction output directory
- `_writePredictionCSV()`: Save predictions to CSV

#### `model.py` - EasyOCRNet

Main model implementation:
- Four-stage architecture (Transformation → Feature Extraction → Sequence Modeling → Prediction)
- Configurable components for each stage
- Supports multiple feature extractors and prediction methods

**Key Methods**:
- `forward(input, text, is_train)`: Forward pass through all stages
- Stages are conditionally executed based on configuration

#### `utilities/feature_extraction.py`

Feature extractor implementations:
- **VGG_FeatureExtractor**: VGG-like CNN with max pooling
- **ResNet_FeatureExtractor**: Custom ResNet with residual blocks
- **RCNN_FeatureExtractor**: Gated RCNN with GRCL layers
- **GRCL**: Gated Recurrent Convolution Layers

#### `utilities/sequence_modeling.py`

Sequence modeling:
- **BidirectionalLSTM**: Bidirectional LSTM for sequence modeling

#### `utilities/prediction.py`

Prediction modules:
- **Attention**: Attention-based sequence-to-sequence prediction
  - **AttentionCell**: Attention mechanism for alignment
  - Teacher forcing for training
  - Autoregressive inference

#### `utilities/transformation.py`

Transformation:
- **TPS_SpatialTransformerNetwork**: Thin-Plate Spline transformation
  - **LocalizationNetwork**: Predicts fiducial points
  - **GridGenerator**: Generates transformation grid
  - Rectifies distorted text images

#### `utilities/labelconverter.py`

Label converters:
- **CTCLabelConverter**: Converts between text and indices for CTC
  - `encode()`: Text → indices (with blank token)
  - `decode_greedy()`: Greedy CTC decoding
  - `decode()`: Generic CTC decoding
- **AttnLabelConverter**: Converts between text and indices for Attention
  - `encode()`: Text → indices (with [GO] and [s] tokens)
  - `decode()`: Attention decoding

### Utils Module

#### `easyocrmetrics.py`

Evaluation metrics:
- **EasyOCRMetricsManager**: WER/CER computation
  - `update()`: Update with predictions and ground truth
  - `compute()`: Compute WER and CER
  - `summarize()`: Return metrics dictionary
  - Uses `jiwer` library for metric computation

## Key Concepts

### PyTorch Lightning

The project uses PyTorch Lightning's abstraction:

- **LightningModule**: Encapsulates model, training, validation, testing logic
- **LightningDataModule**: Encapsulates data loading and preparation
- **Trainer**: Handles training loop, device management, checkpointing
- **Callbacks**: ModelCheckpoint for saving models
- **Loggers**: TensorBoardLogger for metrics visualization

### CTC (Connectionist Temporal Classification)

CTC is a loss function and decoding method for sequence labeling:

- **No alignment needed**: Handles variable-length sequences without explicit alignment
- **Blank token**: Uses blank token (index 0) to handle repeated characters
- **Decoding**: Removes blanks and repeated characters to get final text
- **Loss**: CTCLoss computes alignment-free loss

**CTC Encoding**:
- Text → indices (1 to N), with 0 reserved for blank
- Example: "hello" → [8, 5, 12, 12, 15] (if 'h'=8, 'e'=5, 'l'=12, 'o'=15)

**CTC Decoding**:
- Indices → text (remove blanks and repeats)
- Example: [8, 0, 5, 12, 12, 0, 15] → "hello"

### Attention Mechanism

Attention-based sequence-to-sequence prediction:

- **Encoder-Decoder**: Feature sequence → character sequence
- **Attention**: Aligns encoder features with decoder positions
- **Teacher Forcing**: Uses ground truth during training
- **Autoregressive**: Uses previous predictions during inference

**Attention Encoding**:
- Text → indices with [GO] at start and [s] at end
- Example: "hello" → [[GO], 'h', 'e', 'l', 'l', 'o', [s]]

### TPS (Thin-Plate Spline) Transformation

TPS is used for text rectification:

- **Fiducial Points**: Control points for transformation
- **Localization Network**: Predicts fiducial point locations
- **Grid Generator**: Generates transformation grid
- **Rectification**: Corrects perspective and curvature distortions

### Feature Extraction

Multiple CNN architectures for visual feature extraction:

1. **VGG**: Simple convolutional layers with max pooling
2. **ResNet**: Residual connections for deeper networks
3. **RCNN**: Gated recurrent convolution layers

All extractors output feature maps that are converted to sequences for RNN processing.

### Sequence Modeling

Bidirectional LSTM models sequential dependencies:

- **Bidirectional**: Processes sequence in both directions
- **Context**: Captures left and right context for each position
- **Output**: Contextual feature sequence for prediction

## Workflow

### Training Workflow

1. **Data Preparation**:
   - Organize images and CSV labels
   - Create/update configuration file

2. **Setup**:
   - DataModule sets up train/val datasets
   - ModelModule initializes EasyOCRNet from config
   - Trainer configured with callbacks and logger

3. **Training Loop**:
   - For each epoch:
     - Train on training set
       - Encode labels to indices
       - Forward pass through model
       - Compute loss (CTC or CrossEntropy)
       - Backward pass and optimization
     - Validate on validation set
       - Encode labels
       - Forward pass
       - Decode predictions
       - Compute accuracy, WER, CER
       - Log metrics
     - Save checkpoints (best and last)

4. **Post-Training**:
   - Review metrics in TensorBoard
   - Select best checkpoint for testing

### Testing Workflow

1. **Load Model**: From checkpoint
2. **Run Inference**: On test set
   - Forward pass through model
   - Decode predictions (greedy or beamsearch)
3. **Compute Metrics**: WER, CER using EasyOCRMetricsManager
4. **Save Results**: CSV file with predictions
5. **Log Metrics**: To TensorBoard

### Prediction Workflow

1. **Load Model**: From checkpoint
2. **Load Images**: From specified directory
3. **Run Inference**: 
   - Forward pass through model
   - Decode predictions
4. **Save Results**: CSV file with predictions

## Evaluation Metrics

### Word Error Rate (WER)

WER measures the proportion of words that are incorrectly recognized:

- **Formula**: `WER = (S + D + I) / N`
  - S = substitutions (wrong word)
  - D = deletions (missing word)
  - I = insertions (extra word)
  - N = total words in reference
- **Range**: 0.0 (perfect) to 1.0+ (worse)
- **Lower is better**

### Character Error Rate (CER)

CER measures the proportion of characters that are incorrectly recognized:

- **Formula**: `CER = (S + D + I) / N`
  - S = substitutions (wrong character)
  - D = deletions (missing character)
  - I = insertions (extra character)
  - N = total characters in reference
- **Range**: 0.0 (perfect) to 1.0+ (worse)
- **Lower is better**

### Accuracy

Accuracy measures exact string matching:

- **Formula**: `Accuracy = (Number of exact matches) / (Total samples) × 100`
- **Range**: 0% to 100%
- **Higher is better**

### Metric Computation

The system uses the `jiwer` library for WER and CER computation:
- Computes edit distance (Levenshtein distance)
- Handles word-level (WER) and character-level (CER) errors
- Accumulates metrics across batches
- Summarizes at end of validation epoch

## Tips and Best Practices

### Model Configuration

1. **Feature Extractor**: 
   - VGG: Fast, good for most cases
   - ResNet: Deeper, better for complex text
   - RCNN: Best for challenging cases (more parameters)

2. **Prediction Method**:
   - CTC: Faster, handles variable length well
   - Attention: Better accuracy, more complex

3. **Sequence Modeling**: 
   - BiLSTM: Recommended for better context
   - None: Only for very simple cases

4. **Transformation**:
   - TPS: Useful for distorted text
   - None: For straight text

### Data Configuration

1. **Image Size**: 
   - `imgH: 32`: Standard height for text
   - `imgW: 100`: Adjust based on text width
   - Keep aspect ratio with `PAD: true`

2. **Batch Size**: Adjust based on GPU memory (larger images need smaller batches)

3. **Character Set**: Include all characters in your dataset

### Training Configuration

1. **Learning Rate**: Start with 0.001 for Adam, adjust based on loss
2. **Mixed Precision**: Use `16-mixed` for faster training on GPU
3. **Batch Max Length**: Set based on longest text in dataset
4. **Freezing**: Freeze feature extraction for fine-tuning

### Decoding Configuration

1. **CTC Decoding**:
   - Greedy: Fast, good for most cases
   - Beamsearch: Slower, better accuracy (if implemented)

2. **Attention**: Always uses teacher forcing in training, autoregressive in inference

## Known Issues and Limitations

1. **Beamsearch**: Beamsearch decoding is referenced but may not be fully implemented in labelconverter
2. **Variable Image Sizes**: Images are resized/padded to fixed dimensions (may affect very long text)
3. **Character Set**: Unknown characters are skipped during encoding
4. **TPS**: TPS transformation adds computational overhead

## License

[Add your license information here]

## Authors

[Add author information here]

## Acknowledgments

- CRNN paper authors for the original architecture
- PyTorch Lightning team for the excellent framework
- PyTorch team for the deep learning framework
