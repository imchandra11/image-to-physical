# CraftTraining: CRAFT Text Detection Training Framework

A PyTorch Lightning-based training framework for the CRAFT (Character Region Awareness for Text detection) model. This project provides a modular, extensible system for training text detection models using region and affinity maps with watershed-based post-processing.

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
- [CRAFT Model](#craft-model)
- [Module Documentation](#module-documentation)
- [Key Concepts](#key-concepts)
- [Workflow](#workflow)
- [Post-Processing](#post-processing)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

This project implements a training framework for the **CRAFT** (Character Region Awareness for Text detection) model, which detects text regions in images using region and affinity maps. The system is built on **PyTorch Lightning**, providing a clean abstraction for training, validation, testing, and prediction workflows.

### Key Technologies

- **PyTorch Lightning 2.5.1**: Training framework
- **CRAFT Model**: Character region awareness network for text detection
- **VGG16-BN**: Backbone feature extractor with FPN-like fusion
- **OpenCV**: Image processing and watershed post-processing
- **JSONArgParse**: Configuration management
- **Shapely** (optional): Polygon intersection for evaluation

### What is CRAFT?

CRAFT (Character Region Awareness for Text detection) is a text detection algorithm that:
1. Predicts **region maps** (probability maps for character regions)
2. Predicts **affinity maps** (probability maps for connections between adjacent characters)
3. Uses **watershed algorithm** to segment text regions from the maps
4. Outputs **quadrilateral polygons** for each detected text region

## Project Structure

```
CraftTraining/
│
├── config/                      # Configuration files
│   ├── craft.yaml               # Main configuration file
│   └── craft.local.yaml         # Local overrides (gitignored)
│
├── cli.py                       # Custom Lightning CLI (CRAFTLightningCLI)
├── datamodule.py                # Lightning DataModule for data handling
├── dataset.py                   # CraftDataset - text detection dataset
├── main.py                      # Main entry point for CLI commands
├── main_fittest.py              # Entry point for fit+test workflow
├── model.py                     # CraftNet - CRAFT model architecture
├── module.py                    # CraftLightningModule - Lightning Module for training
└── README.md                    # This file

utils/                           # Shared utility modules (CRAFT-specific)
├── craftTarget.py               # Region/affinity map generation
├── craftmetrics.py              # IoS-based evaluation metrics
├── craft_postprocess.py         # Watershed post-processing
├── craftImageaugmentation.py    # Image augmentation for CRAFT
└── craftImagevisualization.py   # Visualization utilities

CraftTrainer.ipynb               # Jupyter notebook for interactive use
```

## Architecture

### High-Level Architecture

The system follows a **modular, plugin-based architecture**:

1. **Core Framework (CraftTraining/)**: Provides reusable components for CRAFT training
   - CRAFT model implementation (CraftNet)
   - Lightning module for training logic
   - Data module for data loading
   - CLI for command-line interface

2. **Utilities (utils/)**: Shared helper functions
   - Target generation (region/affinity maps)
   - Post-processing (watershed)
   - Metrics computation (IoS-based)
   - Image augmentation and visualization

### Design Patterns

- **Factory Pattern**: Model creation via configuration
- **Template Method Pattern**: Base classes (LightningModule, LightningDataModule) define structure
- **Strategy Pattern**: Different augmentation strategies (training vs validation)
- **Plugin Architecture**: Models and datasets are pluggable via configuration

## Features

### Core Features

1. **CRAFT Model Implementation**
   - VGG16-BN backbone with 5-stage feature extraction
   - FPN-like feature fusion for multi-scale predictions
   - Configurable backbone freezing (partial or full)
   - Pretrained weights support

2. **Text Detection Pipeline**
   - Region and affinity map prediction
   - Watershed-based post-processing
   - Quadrilateral polygon output
   - Configurable thresholds for detection

3. **Complete Training Pipeline**
   - Training with validation
   - Model checkpointing (best and last)
   - Comprehensive evaluation metrics (Precision, Recall, F1)
   - Prediction/inference mode

4. **Image Processing**
   - Custom augmentation pipeline (rotation, scale, crop, flip, color jitter)
   - Gaussian map generation for targets
   - Aspect-ratio preserving resize with padding
   - Polygon-aware transformations

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
- **Learning rate scheduling**: OneCycleLR scheduler

### Evaluation Features

- **IoS-based metrics**: Intersection over Smaller area matching
- **Precision, Recall, F1**: Standard text detection metrics
- **Polygon-based evaluation**: Accurate shape-aware metrics
- **Prediction export**: Text files with polygon coordinates
- **Visualized predictions**: Images with detected text regions

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
- `opencv-python==4.11.0.86`
- `numpy==2.2.2`
- `jsonargparse==4.40.0`
- `tensorboard==2.19.0`
- `shapely` (optional, for accurate polygon intersection metrics)

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
pip install opencv-python numpy jsonargparse tensorboard shapely
```

## Configuration

### Configuration Files

The project uses YAML configuration files with a two-tier system:

1. **Main Config** (`CraftTraining/config/craft.yaml`): Shared configuration
2. **Local Config** (`CraftTraining/config/craft.local.yaml`): Local overrides (should be gitignored)

### Key Configuration Sections

#### Trainer Configuration

```yaml
trainer:
  max_epochs: 50
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 1
  log_every_n_steps: 1
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
  class_path: CraftTraining.module.CraftLightningModule
  init_args:
    save_dir: E:/CraftOutput
    name: CraftTraining
    visualize_training_images: true
    save_predicted_images: true
    pretrained_model: "path/to/pretrained.pth"  # Optional
    ios_threshold: 0.05
    debug_metrics: false
    model:
      class_path: CraftTraining.model.CraftNet
      init_args:
        in_ch: 3
        backbone: vgg16_bn
        pretrained: false
        freeze_until: null  # or "conv2", "conv3", etc.
        feature_extract: false
        head_channels: 128
```

#### Data Configuration

```yaml
data:
  class_path: CraftTraining.datamodule.CraftDataModule
  init_args:
    data_dir: E:/CraftData/data
    batch_size: 3
    num_workers: 0
    resize: 512
    pin_memory: true
    persistent_workers: true
    gauss_cfg:
      gauss_init_size: 200
      gauss_sigma: 10
      enlarge_region: [0.2, 0.2]
      enlarge_affinity: [0.25, 0.25]
      min_sigma: 1.0
    data_cfg:
      custom_aug:
        random_scale:
          range: [1.0, 1.5, 2.0]
          option: false
        random_rotate:
          max_angle: 20
          option: true
        random_crop:
          option: true
          scale: [0.7, 0.9]
        random_horizontal_flip:
          option: true
        random_colorjitter:
          brightness: 0.2
          contrast: 0.2
          saturation: 0.2
          hue: 0.02
          option: true
    test_cfg:
      text_threshold: 0.65
      link_threshold: 0.4
      low_text: 0.3
      vis_opt: true
```

#### Optimizer & Scheduler

```yaml
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.0001
    weight_decay: 0.00001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: 0.001
    total_steps: 50
    pct_start: 0.1
```

### Creating Local Configuration

Create `CraftTraining/config/craft.local.yaml` to override settings:

```yaml
trainer:
  max_epochs: 10
  limit_train_batches: 100  # For testing

model:
  init_args:
    visualize_training_images: true

data:
  init_args:
    data_dir: /path/to/your/data
    batch_size: 2
```

## Usage

### Data Directory Structure

Your data should be organized as follows:

```
data_dir/
├── Training/
│   ├── Input/          # Training images
│   └── Output/         # Ground truth text files (gt_*.txt)
├── Validation/
│   ├── Input/
│   └── Output/
└── Testing/
    ├── Input/
    └── Output/
```

**Note**: If `Input/` folder doesn't exist, images are expected in the root of each split folder.

### Ground Truth Text Format

Each image should have a corresponding text file in the `Output/` folder named `gt_<imagename>.txt` with the following format:

```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
x1,y1,x2,y2,x3,y3,x4,y4,transcription
...
```

Each line represents one text region as a quadrilateral polygon with 4 points (x, y coordinates) and an optional transcription.

**Example:**
```
100,50,200,50,200,100,100,100,"Hello"
250,75,350,75,350,125,250,125,"World"
```

### Training

#### Using CLI (Recommended)

```bash
python CraftTraining/main.py fit \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml
```

#### Using main_fittest.py (Fit + Test)

```bash
python CraftTraining/main_fittest.py \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml
```

#### Using Jupyter Notebook

See `CraftTrainer.ipynb` for interactive usage:

```python
from CraftTraining.module import CraftLightningModule
from CraftTraining.datamodule import CraftDataModule

%run CraftTraining/main_fittest.py \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml
```

### Resuming Training

To resume from a checkpoint:

```bash
python CraftTraining/main.py fit \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml \
  --fit.ckpt_path path/to/checkpoint.ckpt
```

Or add to config:

```yaml
fit:
  ckpt_path: path/to/checkpoint.ckpt
```

### Testing

```bash
python CraftTraining/main.py test \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml \
  --test.ckpt_path best  # or 'last' or path to checkpoint
```

### Prediction/Inference

```bash
python CraftTraining/main.py predict \
  --config CraftTraining/config/craft.yaml \
  --config CraftTraining/config/craft.local.yaml \
  --data.data_dir /path/to/test/images \
  --test.ckpt_path path/to/checkpoint.ckpt
```

**Note**: For prediction, results are saved in the `prediction/` folder within the log directory. Each image produces:
- A `.txt` file with detected polygons and scores
- A visualized image (if `save_predicted_images: true`)

## Data Format

### Input Images

- **Supported formats**: PNG, JPG, JPEG, TIF, TIFF, BMP
- **Processing pipeline**: 
  1. Read image (BGR format via OpenCV)
  2. Apply augmentations (training only)
  3. Convert to grayscale
  4. Resize keeping aspect ratio
  5. Center pad to square canvas
  6. Normalize to [0, 1]

### Ground Truth Format

Text files (`gt_*.txt`) with format:
- Each line: `x1,y1,x2,y2,x3,y3,x4,y4,transcription`
- Coordinates are quadrilateral polygon vertices (4 points)
- Transcription is optional (can be empty or quoted string)
- Coordinates are in pixel coordinates relative to the original image

### Output Format

#### Training/Validation

- Loss values logged to TensorBoard
- Checkpoints saved automatically

#### Testing

- **Text files**: One per image with detections
  - Format: `x1,y1,x2,y2,x3,y3,x4,y4,score`
  - Coordinates are in original image space
  - Score is the mean region score
- **Images**: Visualized predictions with polygon overlays
- **Metrics**: Precision, Recall, F1 logged to TensorBoard

#### Prediction

- Same as testing, but without metrics computation

## CRAFT Model

### Model Architecture

The CRAFT model (`CraftNet`) consists of:

1. **Backbone (VGG16-BN)**: Feature extraction in 5 stages
   - Stage 1: Conv1_1, Conv1_2, Pool (64 channels)
   - Stage 2: Conv2_1, Conv2_2, Pool (128 channels)
   - Stage 3: Conv3_1, Conv3_2, Conv3_3, Pool (256 channels)
   - Stage 4: Conv4_1, Conv4_2, Conv4_3, Pool (512 channels)
   - Stage 5: Conv5_1, Conv5_2, Conv5_3, Pool (512 channels)

2. **Feature Fusion (FPN-like)**: Multi-scale feature fusion
   - Lateral convolutions to reduce channel dimensions
   - Top-down feature merging (stage5 → stage4 → stage3 → stage2 → stage1)
   - Bilinear upsampling and concatenation

3. **Prediction Head**: Region and affinity map prediction
   - Final convolution to produce 2-channel output
   - Region logit (character regions)
   - Affinity logit (character connections)

### Model Features

- **Configurable Backbone Freezing**: Freeze early stages for fine-tuning
- **Pretrained Support**: Load ImageNet or CRAFT pretrained weights
- **Flexible Input Channels**: Supports 1-channel (grayscale) or 3-channel input
- **Multi-scale Predictions**: FPN-like fusion for better detection of varying text sizes

### Freezing Options

The model supports flexible backbone freezing:

- `freeze_until: null` or `"none"`: No freezing (train all layers)
- `freeze_until: "conv2"`: Freeze stage 1
- `freeze_until: "conv3"`: Freeze stages 1 and 2
- `feature_extract: true`: Freeze entire backbone (only head trains)

## Module Documentation

### CraftTraining Module

#### `cli.py` - CRAFTLightningCLI

Custom CLI class extending LightningCLI with:
- Checkpoint path arguments for fit and test

#### `datamodule.py` - CraftDataModule

PyTorch Lightning DataModule that:
- Manages train/val/test/predict datasets
- Handles data loading with proper transforms
- Supports custom collate function for variable-size images
- Provides data loaders with padding for batch processing

**Key Methods**:
- `setup(stage)`: Prepares datasets for different stages
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `predict_dataloader()`
- `transfer_batch_to_device()`: Custom device transfer for padded batches
- `_collate_fn()`: Custom collate function for variable-size images (padding to max size in batch)

#### `dataset.py` - CraftDataset

Dataset class for text detection:
- Loads images from Input folder
- Finds corresponding text annotations in Output folder
- Applies augmentations (rotation, scale, crop, flip, color jitter)
- Generates region and affinity maps from polygon annotations
- Returns images, targets (maps + metadata), and image names

**Key Methods**:
- `__getitem__(idx)`: Returns image tensor, target dict, and image name
- `_read_label_file()`: Parses ground truth text files
- `_resize_keep_aspect_and_pad()`: Resize with aspect ratio preservation

#### `module.py` - CraftLightningModule

PyTorch Lightning Module for CRAFT training:
- Wraps CRAFT model (CraftNet)
- Handles training, validation, testing, and prediction
- Computes BCE loss for region and affinity maps
- Performs watershed post-processing for inference
- Computes IoS-based metrics
- Saves predictions and visualizations

**Key Methods**:
- `training_step()`: Forward pass and loss computation
- `validation_step()`: Validation with loss
- `test_step()`: Inference with metrics
- `predict_step()`: Inference without metrics
- `_compute_loss()`: BCE loss for region + affinity maps
- `_runInference()`: Core inference logic with watershed post-processing
- `_preparePrediction()`: Setup prediction output directory
- `_load_test_cfg()`: Load test thresholds from config

#### `model.py` - CraftNet

CRAFT model implementation:
- VGG16-BN backbone with 5-stage feature extraction
- FPN-like feature fusion
- Prediction head for region and affinity maps
- Configurable freezing and pretrained weights

**Key Methods**:
- `forward(x)`: Forward pass returning region and affinity logits
- `_parse_conv_block_index()`: Parse freeze_until string
- `_ensure_input_channels()`: Handle variable input channels
- `count_params()`: Count total and trainable parameters

#### `main.py` - CLI Entry Point

Standard Lightning CLI entry point supporting:
- `fit`: Training
- `test`: Testing
- `predict`: Inference

#### `main_fittest.py` - Fit+Test Entry Point

Runs training followed by testing in one command.

### Utils Module (CRAFT-specific)

#### `craftTarget.py`

Target generation utilities:
- `generate_region_affinity_maps()`: Generates region and affinity maps from polygon annotations
  - Region maps: Gaussian-blurred character regions
  - Affinity maps: Connections between adjacent characters in words
  - Configurable Gaussian parameters and enlargement factors

#### `craftmetrics.py`

Evaluation metrics:
- `CraftMetrics`: IoS-based text detection metrics
  - `compute_ios()`: Intersection over Smaller area between polygons
  - `update()`: Update metrics with predictions and ground truth
  - `compute()`: Compute Precision, Recall, F1

#### `craft_postprocess.py`

Post-processing utilities:
- `craft_watershed()`: Watershed-based text region extraction from score maps
  - Threshold region and affinity maps
  - Create combined mask
  - Distance transform for seed generation
  - Watershed segmentation
  - Quadrilateral polygon fitting
  - Score computation

#### `craftImageaugmentation.py`

Image augmentation:
- `get_transform()`: Build augmentation pipeline from config
  - Random scale
  - Random rotation
  - Random crop (keeping polygons)
  - Random horizontal flip
  - Color jitter (brightness, contrast, saturation, hue)
- Polygon-aware transformations (all augmentations transform polygons accordingly)

#### `craftImagevisualization.py`

Visualization utilities:
- `visualizeOneBatchImages()`: Visualize batch of images with polygon annotations
- `drawCV2BBWithText()`: Draw bounding boxes with text labels
- `save_prediction_visual()`: Save visualized predictions

## Key Concepts

### PyTorch Lightning

The project uses PyTorch Lightning's abstraction:

- **LightningModule**: Encapsulates model, training, validation, testing logic
- **LightningDataModule**: Encapsulates data loading and preparation
- **Trainer**: Handles training loop, device management, checkpointing
- **Callbacks**: ModelCheckpoint for saving models
- **Loggers**: TensorBoardLogger for metrics visualization

### CRAFT Detection Format

The system uses CRAFT's detection format:

**Target Dictionary** (training):
```python
{
    'region': tensor(H, W),      # Region map (character regions)
    'affinity': tensor(H, W),    # Affinity map (character connections)
    'polys': List[np.ndarray],   # Polygon annotations
    'boxes': tensor(N, 4),       # Axis-aligned bounding boxes
    'orig_size': (W, H),         # Original image size
    'curr_size': (W, H),         # Current (resized) size
    'scale': float,              # Scale factor
    'pad_offset': (x, y),        # Padding offset
    ...
}
```

**Model Output**:
```python
{
    'region_logit': tensor(B, 1, H, W),   # Region logits
    'affinity_logit': tensor(B, 1, H, W)  # Affinity logits
}
```

**Post-processed Output**:
- List of polygons (4-point quadrilaterals) with scores

### Region and Affinity Maps

- **Region Map**: Probability map indicating character regions
  - Generated from character polygon annotations
  - Gaussian-blurred for smooth probability distribution
  - Used to detect individual characters

- **Affinity Map**: Probability map indicating connections between characters
  - Generated from adjacent character pairs within words
  - Links characters that belong to the same word
  - Used to group characters into words

### Watershed Post-Processing

The watershed algorithm is used to extract text regions from score maps:

1. **Threshold**: Create binary masks from region/affinity maps
2. **Seed Generation**: Use distance transform to find text cores
3. **Watershed Segmentation**: Segment regions from seeds
4. **Polygon Fitting**: Convert segments to quadrilateral polygons
5. **Filtering**: Remove small regions and compute scores

### IoS-based Metrics

Intersection over Smaller (IoS) is used for polygon matching:

- IoS = Intersection Area / min(Predicted Area, Ground Truth Area)
- More lenient than IoU for text detection
- Better handles overlapping text regions
- Threshold typically set to 0.05-0.5

## Workflow

### Training Workflow

1. **Data Preparation**:
   - Organize images and text annotations
   - Create/update configuration files

2. **Setup**:
   - DataModule sets up train/val datasets
   - ModelModule initializes CraftNet from config
   - Trainer configured with callbacks and logger

3. **Training Loop**:
   - For each epoch:
     - Train on training set (with augmentations)
     - Validate on validation set
     - Log metrics to TensorBoard
     - Save checkpoints (best and last)

4. **Post-Training**:
   - Review metrics in TensorBoard
   - Select best checkpoint for testing

### Testing Workflow

1. **Load Model**: From checkpoint
2. **Run Inference**: On test set (no augmentations)
3. **Post-process**: Watershed segmentation to extract polygons
4. **Compute Metrics**: IoS-based Precision, Recall, F1
5. **Save Results**: Text files, visualized images
6. **Log Metrics**: To TensorBoard

### Prediction Workflow

1. **Load Model**: From checkpoint
2. **Load Images**: From specified directory
3. **Run Inference**: Generate region/affinity maps
4. **Post-process**: Watershed to extract polygons
5. **Filter by Thresholds**: Remove low-confidence detections
6. **Save Results**: Text files, visualized images

## Post-Processing

### Watershed Algorithm

The watershed post-processing extracts text regions:

1. **Threshold Maps**:
   - `text_threshold`: Threshold for region map (text cores)
   - `link_threshold`: Threshold for affinity map (connections)
   - `low_text`: Lower threshold for combined mask

2. **Seed Generation**:
   - Distance transform on text cores
   - Peak detection for seed regions
   - Morphological operations to connect nearby seeds

3. **Segmentation**:
   - Combined mask (low_text OR link)
   - Watershed algorithm to segment regions
   - Each region corresponds to a text instance

4. **Polygon Fitting**:
   - Contour extraction from segments
   - Approximate polygon to 4-point quadrilateral
   - Order points clockwise

5. **Filtering**:
   - Remove small regions (min_area)
   - Compute confidence score (mean region score)
   - Scale polygons to original image coordinates

### Configuration

Post-processing thresholds can be configured in `data.test_cfg`:

```yaml
test_cfg:
  text_threshold: 0.65    # Region map threshold (higher = fewer detections)
  link_threshold: 0.4     # Affinity map threshold
  low_text: 0.3           # Lower threshold for combined mask
  min_area: 10            # Minimum region area (pixels)
  vis_opt: true           # Enable visualization
```

## Evaluation Metrics

### IoS-based Metrics

The evaluation uses Intersection over Smaller (IoS) for matching:

- **IoS Calculation**: `IoS = Intersection Area / min(Predicted Area, Ground Truth Area)`
- **Matching**: Prediction matches GT if IoS > threshold (default: 0.05)
- **One-to-Many Matching**: One prediction can match multiple GTs
- **Metrics**: Precision, Recall, F1 (harmonic mean)

### Metric Definitions

- **True Positive (TP)**: Number of GT polygons matched by predictions
- **False Positive (FP)**: Number of predictions not matched to any GT
- **False Negative (FN)**: Number of GT polygons not matched by any prediction
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1 (hmean)**: `2 * Precision * Recall / (Precision + Recall)`

### Advantages of IoS

- More lenient than IoU (Intersection over Union)
- Better for overlapping text regions
- Suitable for text detection where partial matches are meaningful
- Robust to polygon shape variations

## Tips and Best Practices

### Model Configuration

1. **Backbone Freezing**: Start with `freeze_until: "conv2"` or `"conv3"` for fine-tuning
2. **Pretrained Weights**: Use CRAFT pretrained weights if available
3. **Head Channels**: Default 128 works well; increase for larger images
4. **Feature Extract**: Set to `true` only for feature extraction tasks

### Data Configuration

1. **Batch Size**: Adjust based on GPU memory (larger images need smaller batches)
2. **Resize**: 512 is a good default; increase for higher resolution text
3. **Augmentation**: Enable rotation and color jitter for robustness
4. **Gaussian Config**: Adjust sigma based on character size in your data

### Training Configuration

1. **Learning Rate**: Start with 0.0001 and adjust based on loss
2. **Mixed Precision**: Use `16-mixed` for faster training on GPU
3. **Validation**: Monitor validation loss to prevent overfitting
4. **Checkpointing**: Save both best and last checkpoints

### Post-Processing

1. **Thresholds**: Tune `text_threshold` and `link_threshold` based on your data
2. **Min Area**: Filter small false positives with appropriate `min_area`
3. **Visualization**: Enable `vis_opt: true` to debug detection issues

## Known Issues and Limitations

1. **Single Backbone**: Currently only supports VGG16-BN (other backbones can be added)
2. **Fixed Input Size**: Model expects square input (padding handles aspect ratio)
3. **Watershed Limitations**: May struggle with very close or overlapping text
4. **IoS Metrics**: Requires Shapely for accurate polygon intersection (falls back to AABB if not available)

## License

[Add your license information here]

## Authors

[Add author information here]

## Acknowledgments

- CRAFT paper authors for the original algorithm
- PyTorch Lightning team for the excellent framework
- PyTorch team for the deep learning framework
