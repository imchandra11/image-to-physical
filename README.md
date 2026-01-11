# Image-to-Physical: Object Detection for View Recognition

A PyTorch Lightning-based object detection system specifically designed for recognizing views, title blocks, and BOM (Bill of Materials) tables in engineering drawings and technical images. This project provides a modular, extensible framework for object detection tasks with support for custom datasets and models.

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
- [Module Documentation](#module-documentation)
- [Key Concepts](#key-concepts)
- [Workflow](#workflow)
- [Known Issues](#known-issues)

## Overview

This project implements an object detection system using **Faster R-CNN** with a **ResNet-50 FPN backbone** for detecting and localizing specific regions in technical drawings. The primary use case is **View Recognition** - identifying main views, title blocks, and BOM tables in engineering drawings.

The system is built on **PyTorch Lightning**, which provides a clean abstraction for training, validation, testing, and prediction workflows. It uses **JSONArgParse** (via Lightning CLI) for flexible configuration management through YAML files.

### Key Technologies

- **PyTorch Lightning 2.5.1**: Training framework
- **Faster R-CNN**: Object detection model architecture
- **ResNet-50 FPN**: Backbone feature extractor
- **Albumentations**: Image augmentation library
- **OpenCV**: Image processing
- **JSONArgParse**: Configuration management
- **TorchMetrics**: Model evaluation metrics (mAP)

## Project Structure

```
image-to-physical/
│
├── ObjectDetection/           # Core object detection framework
│   ├── cli.py                 # Custom Lightning CLI with extensions
│   ├── datamodule.py          # Lightning DataModule for data handling
│   ├── dataset.py             # Base dataset classes (DatasetOD, DatasetImage)
│   ├── main.py                # Main entry point for CLI commands
│   ├── main_fittest.py        # Entry point for fit+test workflow
│   ├── modelmodule.py         # Lightning Module for object detection model
│   └── modelfactory.py        # Factory functions for creating models
│
├── ViewRecognition/           # View Recognition specific implementation
│   ├── config/
│   │   ├── viewrecognition.yaml          # Main configuration file
│   │   └── viewrecognition.local.yaml    # Local overrides (gitignored)
│   └── dataset.py             # DatasetVR - View Recognition dataset
│
├── utils/                     # Shared utility modules
│   ├── colors.py              # Color palette utilities
│   ├── dataset.py             # Collate function for batching
│   ├── image.py               # Image processing utilities
│   ├── imageaugmentation.py   # Image augmentation functions
│   ├── imagevisualization.py  # Visualization functions
│   ├── lib.py                 # Configuration utility (getCallableAndArgs)
│   └── metrics.py             # Evaluation metrics (IoU, confusion matrix)
│
├── ViewRecognition.ipynb      # Jupyter notebook for interactive use
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Architecture

### High-Level Architecture

The system follows a **modular, plugin-based architecture**:

1. **Core Framework (ObjectDetection/)**: Provides reusable components for any object detection task
   - Generic dataset classes
   - Lightning modules for training/validation/test
   - Data module for data loading
   - CLI for command-line interface

2. **Application Layer (ViewRecognition/)**: Implements specific use case
   - Custom dataset implementation (DatasetVR)
   - Application-specific configuration

3. **Utilities (utils/)**: Shared helper functions
   - Image processing
   - Visualization
   - Metrics computation

### Design Patterns

- **Factory Pattern**: Model and dataset creation via factory functions/config
- **Template Method Pattern**: Base classes (DatasetOD, LightningModule) define structure
- **Strategy Pattern**: Different augmentation strategies (getTransform vs getNoTransform)
- **Plugin Architecture**: Models and datasets are pluggable via configuration

## Features

### Core Features

1. **Modular Object Detection Framework**
   - Reusable base classes for datasets and models
   - Easy to extend for new object detection tasks
   - Pluggable models and datasets via configuration

2. **View Recognition Specific**
   - Detects three classes: Main views, Title blocks, BOM tables
   - Processes engineering drawings with CSV annotations
   - Handles grayscale binary images

3. **Complete Training Pipeline**
   - Training with validation
   - Model checkpointing (best and last)
   - Comprehensive evaluation metrics (mAP, confusion matrix)
   - Prediction/inference mode

4. **Image Processing**
   - Preprocessing: Color → Grayscale → Binary → Resize
   - Data augmentation for training (rotation, cropping, noise, etc.)
   - Visualization tools for debugging

5. **Flexible Configuration**
   - YAML-based configuration
   - Local config overrides
   - Command-line argument support
   - Automatic linking of model/dataset parameters

### Training Features

- **Automatic mixed precision** (if supported)
- **Checkpoint management**: Best and last checkpoints
- **TensorBoard logging**: Metrics, images, confusion matrices
- **Validation during training**: Per-epoch validation
- **Early stopping** (configurable via callbacks)
- **Learning rate scheduling**: OneCycleLR scheduler

### Evaluation Features

- **Mean Average Precision (mAP)**: Standard object detection metric
- **Per-class metrics**: Precision, recall per class
- **Confusion Matrix**: True positives, false positives, false negatives
- **Prediction Export**: CSV files with bounding boxes and scores
- **Visualized Predictions**: Images with drawn bounding boxes

## Requirements

### System Requirements

- **Python**: 3.12.10
- **CUDA**: 12.6.3 (for GPU support)
- **Operating System**: Windows/Linux/MacOS

### Python Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `torch==2.6.0+cu126`
- `torchvision==0.21.0+cu126`
- `lightning==2.5.1.post0` (PyTorch Lightning)
- `albumentations==1.3.1`
- `opencv-python==4.11.0.86`
- `pandas==2.2.3`
- `jsonargparse==4.40.0`
- `torchmetrics==1.7.1`
- `tensorboard==2.19.0`

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
pip install opencv-python pandas albumentations==1.3.1 jsonargparse tensorboard torchmetrics[detection] tabulate seaborn
```

## Configuration

### Configuration Files

The project uses YAML configuration files with a two-tier system:

1. **Main Config** (`ViewRecognition/config/viewrecognition.yaml`): Shared configuration
2. **Local Config** (`ViewRecognition/config/viewrecognition.local.yaml`): Local overrides (should be gitignored)

### Key Configuration Sections

#### Trainer Configuration

```yaml
trainer:
  max_epochs: 100
  num_sanity_val_steps: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 1
  callbacks:
    - ModelCheckpoint for best model
    - ModelCheckpoint for last model
  logger:
    TensorBoardLogger configuration
```

#### Model Configuration

```yaml
model:
  class_path: ObjectDetection.modelmodule.ModelModuleOD
  init_args:
    detection_threshold: 0.5
    visualize_training_images: false
    save_predicted_images: true
    torch_model_factory:
      function_path: ObjectDetection.modelfactory.getModelfasterrcnn_resnet50_fpn
      init_args:
        pretrained: true
```

#### Data Configuration

```yaml
data:
  class_path: ObjectDetection.datamodule.DataModuleOD
  init_args:
    data_dir: E:\AIData\Bbox_Dataset
    batch_size: 2
    num_workers: 4
    resize: 600
    classes:
      background: __background__
      Main: 1
      Title: 2
      BOM: 3
    dataset:
      class_path: ViewRecognition.dataset.DatasetVR
```

#### Optimizer & Scheduler

```yaml
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
    weight_decay: 0.00001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: 0.001
    pct_start: 0.1
    total_steps: 100
```

### Creating Local Configuration

Create `ViewRecognition/config/viewrecognition.local.yaml` to override settings:

```yaml
trainer:
  max_epochs: 10
  limit_train_batches: 100  # For testing
data:
  init_args:
    data_dir: /path/to/your/data
model:
  init_args:
    detection_threshold: 0.3
```

## Usage

### Data Directory Structure

Your data should be organized as follows:

```
data_dir/
├── Training/
│   ├── Input/          # Training images
│   └── Output/         # CSV annotation files
├── Validation/
│   ├── Input/
│   └── Output/
└── Testing/
    ├── Input/
    └── Output/
```

**Note**: If `Input/` folder doesn't exist, images are expected in the root of each split folder.

### CSV Annotation Format

Each image should have a corresponding CSV file in the `Output/` folder with the following columns:

```csv
MainLabel,SubLabel,MinX,MinY,MaxX,MaxY,CenterX,CenterY,Width,Height,Value
View,MainView,100,200,300,400,200,300,200,200,...
View,FrameTitleBlock,50,50,150,100,100,75,100,50,...
View,BOMTable,400,100,600,500,500,300,200,400,...
```

### Training

#### Using CLI (Recommended)

```bash
python ObjectDetection/main.py fit \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml
```

#### Using main_fittest.py (Fit + Test)

```bash
python ObjectDetection/main_fittest.py \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml
```

#### Using Jupyter Notebook

See `ViewRecognition.ipynb` for interactive usage:

```python
from ObjectDetection.modelmodule import ModelModuleOD
from ObjectDetection.datamodule import DataModuleOD

%run ObjectDetection/main_fittest.py \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml
```

### Resuming Training

To resume from a checkpoint:

```bash
python ObjectDetection/main.py fit \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml \
  --fit.ckpt_path path/to/checkpoint.ckpt
```

Or add to config:

```yaml
fit:
  ckpt_path: path/to/checkpoint.ckpt
```

### Testing

```bash
python ObjectDetection/main.py test \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml \
  --test.ckpt_path best  # or 'last' or path to checkpoint
```

### Prediction/Inference

```bash
python ObjectDetection/main.py predict \
  --config ViewRecognition/config/viewrecognition.yaml \
  --config ViewRecognition/config/viewrecognition.local.yaml \
  --data.data_dir /path/to/test/images \
  --test.ckpt_path path/to/checkpoint.ckpt
```

**Note**: For prediction, images can be in a single directory (no Input/Output structure needed). Results are saved in the `prediction/` folder within the log directory.

## Data Format

### Input Images

- **Supported formats**: PNG, JPG, JPEG, TIF
- **Processing pipeline**: 
  1. Read image (BGR format via OpenCV)
  2. Convert to RGB
  3. Convert to grayscale (mean of RGB channels)
  4. Binary threshold (threshold=50)
  5. Resize (default 600x600)
  6. Normalize (divide by max, so max=1)

### Annotation CSV Format

Required columns:

- `MainLabel`: Main category (e.g., "View")
- `SubLabel`: Subcategory (e.g., "MainView", "FrameTitleBlock", "BOMTable")
- `MinX, MinY, MaxX, MaxY`: Bounding box coordinates (original image scale)
- `CenterX, CenterY`: Center coordinates
- `Width, Height`: Bounding box dimensions
- `Value`: Additional value (may be empty)

### Label Mapping

The View Recognition dataset maps labels as follows:

- `MainView` → Class 1 (Main)
- `FrameTitleBlock` → Class 2 (Title)
- `BOMTable` → Class 3 (BOM)

Background class (0) is automatically added for object detection models.

### Output Format

#### Training/Validation

- Loss values logged to TensorBoard
- Checkpoints saved automatically

#### Testing

- **CSV files**: One per image with detections
  - Columns: Left, Top, Right, Bot, Score, Label
- **Images**: Visualized predictions with bounding boxes
- **Metrics**: mAP, confusion matrix logged to TensorBoard

#### Prediction

- Same as testing, but without metrics computation

## Module Documentation

### ObjectDetection Module

#### `cli.py` - ODLightningCLI

Custom CLI class extending LightningCLI with:
- Checkpoint path arguments for fit and test
- Automatic linking of dataset classes to model num_classes

#### `datamodule.py` - DataModuleOD

PyTorch Lightning DataModule that:
- Manages train/val/test/predict datasets
- Handles data loading with proper transforms
- Supports custom dataset classes via configuration
- Provides data loaders with custom collate function

**Key Methods**:
- `setup(stage)`: Prepares datasets for different stages
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `predict_dataloader()`
- `transfer_batch_to_device()`: Custom device transfer for variable-length batches

#### `dataset.py` - DatasetOD, DatasetImage

**DatasetOD**: Base dataset class for object detection
- Loads images from Input folder
- Finds corresponding CSV annotations in Output folder
- Must be subclassed to implement `__getitem__()`

**DatasetImage**: Dataset for prediction (no annotations)
- Loads images only
- Returns empty target dictionaries

#### `modelmodule.py` - ModelModuleOD

PyTorch Lightning Module for object detection:
- Wraps PyTorch object detection model (e.g., Faster R-CNN)
- Handles training, validation, testing, and prediction
- Computes metrics (mAP, confusion matrix)
- Saves predictions and visualizations

**Key Methods**:
- `training_step()`: Forward pass and loss computation
- `validation_step()`: Validation with loss
- `test_step()`: Inference with metrics
- `predict_step()`: Inference without metrics
- `_runInference()`: Core inference logic
- `_preparePrediction()`: Setup prediction output directory

#### `modelfactory.py` - Model Factory Functions

Factory functions for creating models:
- `getModelfasterrcnn_resnet50_fpn()`: Creates Faster R-CNN with ResNet-50 FPN backbone
  - Supports pretrained weights
  - Replaces predictor head for custom number of classes
  - Adds dropout to backbone

#### `main.py` - CLI Entry Point

Standard Lightning CLI entry point supporting:
- `fit`: Training
- `test`: Testing
- `predict`: Inference

#### `main_fittest.py` - Fit+Test Entry Point

Runs training followed by testing in one command.

### ViewRecognition Module

#### `dataset.py` - DatasetVR

View Recognition specific dataset implementation:
- Extends `DatasetOD`
- Reads CSV annotations
- Filters for "View" MainLabel
- Maps SubLabels to classes (MainView→Main, FrameTitleBlock→Title, BOMTable→BOM)
- Converts center/width/height format to bounding boxes
- Applies transforms (augmentation or no-transform)

### Utils Module

#### `colors.py`

Color utilities:
- `getRandomBASEColors()`: Returns random base colors (RGB 0-255)
- `getRandomTABLEAUColors()`: Returns Tableau color palette (HEX)
- `hex_to_bgr()`: Converts HEX to BGR for OpenCV

#### `dataset.py`

- `collate_fn()`: Custom collate function for variable-length batches

#### `image.py`

- `convertToBoundingBox()`: Converts center/width/height to corner coordinates with margins

#### `imageaugmentation.py`

Image preprocessing and augmentation:
- `preProcess()`: Color → Grayscale → Binary → Resize → Normalize
- `getTransform()`: Training augmentations (crop, rotate, blur, noise, invert, color jitter)
- `getNoTransform()`: Test-time transform (ToTensor only)

#### `imagevisualization.py`

Visualization utilities:
- `drawCV2BBWithText()`: Draw bounding box with label
- `visualizeOneBatchImages()`: Visualize batch of images
- `visualizeImage()`: Visualize single image with targets

#### `lib.py`

Configuration utility functions:
- `getCallableAndArgs()`: Extracts callable objects (classes or functions) and their initialization arguments from configuration dictionaries. Used by `ModelModuleOD` and `DataModuleOD` to dynamically instantiate models and datasets from YAML configuration.
- `getAttr()`: Helper function to dynamically import and get attributes from modules using dot-notation paths (e.g., "ObjectDetection.modelfactory.getModelfasterrcnn_resnet50_fpn").

#### `metrics.py`

Evaluation metrics:
- `computeIOU()`: Intersection over Union for bounding boxes
- `computeBBConfusionMatrix()`: Confusion matrix for object detection (TP, FP, FN per class)

## Key Concepts

### PyTorch Lightning

The project uses PyTorch Lightning's abstraction:

- **LightningModule**: Encapsulates model, training, validation, testing logic
- **LightningDataModule**: Encapsulates data loading and preparation
- **Trainer**: Handles training loop, device management, checkpointing
- **Callbacks**: ModelCheckpoint for saving models
- **Loggers**: TensorBoardLogger for metrics visualization

### Object Detection Format

The system uses the standard PyTorch object detection format:

**Target Dictionary**:
```python
{
    'boxes': tensor([[x1, y1, x2, y2], ...]),  # Pascal VOC format
    'labels': tensor([class_id, ...]),
    'area': tensor([area, ...]),
    'iscrowd': tensor([0, 0, ...]),
    'image_id': tensor([idx])
}
```

**Model Output**:
```python
[
    {
        'boxes': tensor([[x1, y1, x2, y2], ...]),
        'scores': tensor([confidence, ...]),
        'labels': tensor([class_id, ...])
    },
    ...
]
```

### Configuration-Driven Design

Models and datasets are instantiated from configuration:

```yaml
torch_model_factory:
  function_path: ObjectDetection.modelfactory.getModelfasterrcnn_resnet50_fpn
  init_args:
    pretrained: true
```

This allows switching models/datasets without code changes.

### Image Processing Pipeline

1. **Preprocessing** (`preProcess`):
   - BGR → RGB → Grayscale → Binary threshold → Resize → Normalize

2. **Training Augmentation** (`getTransform`):
   - Crop and pad
   - Random crop from borders
   - Random 90° rotation
   - Blur, Gaussian noise
   - Invert
   - Color jitter
   - ToTensor

3. **Test/Inference** (`getNoTransform`):
   - ToTensor only

## Workflow

### Training Workflow

1. **Data Preparation**:
   - Organize images and CSV annotations
   - Create/update configuration files

2. **Setup**:
   - DataModule sets up train/val datasets
   - ModelModule initializes model from config
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
3. **Compute Metrics**: mAP, confusion matrix
4. **Save Results**: CSV files, visualized images
5. **Log Metrics**: To TensorBoard

### Prediction Workflow

1. **Load Model**: From checkpoint
2. **Load Images**: From specified directory
3. **Run Inference**: Generate predictions
4. **Filter by Threshold**: Remove low-confidence detections
5. **Save Results**: CSV files, visualized images

## Potential Improvements

1. **Error Handling**: Add more validation and error messages
2. **Documentation**: Add docstrings to all functions
3. **Testing**: Add unit tests
4. **Type Hints**: Add complete type annotations
5. **Configuration Validation**: Validate config before training
6. **Data Validation**: Validate data format before processing

## License

[Add your license information here]

## Authors

[Add author information here]

## Acknowledgments

- PyTorch Lightning team for the excellent framework
- PyTorch team for the deep learning framework
- Albumentations for image augmentation utilities
