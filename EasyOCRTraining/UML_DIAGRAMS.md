# UML Diagrams and Design Documentation - EasyOCRTraining

This document contains comprehensive UML diagrams, class diagrams, sequence diagrams, High-Level Design (HLD), Low-Level Design (LLD), EasyOCR model architecture, and visual representations of the project structure and interconnections.

## Table of Contents

- [High-Level Design (HLD)](#high-level-design-hld)
- [Low-Level Design (LLD)](#low-level-design-lld)
- [Class Diagrams](#class-diagrams)
- [EasyOCR Model Architecture](#easyocr-model-architecture)
- [Sequence Diagrams](#sequence-diagrams)
- [Folder and File Structure](#folder-and-file-structure)
- [Module Interconnections](#module-interconnections)
- [Data Flow Diagrams](#data-flow-diagrams)

---

## High-Level Design (HLD)

### System Overview

```mermaid
flowchart TD
    System["EasyOCRTraining System<br/>OCR Text Recognition Framework"]
    
    Config["Configuration<br/>Management"]
    Training["Training<br/>Pipeline"]
    Inference["Inference<br/>Pipeline"]
    
    PyTorchLightning["PyTorch<br/>Lightning<br/>Framework"]
    ModelEngine["EasyOCR Model<br/>Inference Engine"]
    
    DataModule["Data<br/>Module"]
    ModelModule["Model<br/>Module"]
    UtilsModule["Utils<br/>Modules"]
    
    OCRDataset["OCRDataset<br/>Dataset"]
    ConfigFiles["Config<br/>Files"]
    Notebook["Notebook<br/>Interface"]
    
    System --> Config
    System --> Training
    System --> Inference
    
    Training --> PyTorchLightning
    Inference --> ModelEngine
    
    Config --> DataModule
    Config --> ModelModule
    Config --> UtilsModule
    
    DataModule --> OCRDataset
    ModelModule --> OCRDataset
    UtilsModule --> OCRDataset
    
    ConfigFiles --> OCRDataset
    Notebook --> OCRDataset
    
    style System fill:#e1f5ff
    style DataModule fill:#fff4e1
    style ModelModule fill:#fff4e1
    style UtilsModule fill:#fff4e1
    style OCRDataset fill:#e8f5e9
```

### Component Architecture

```mermaid
flowchart TD
    UILayer["User Interface Layer<br/>• CLI (Command Line Interface)<br/>• Jupyter Notebook<br/>• Configuration Files (YAML)"]
    
    OrchestrationLayer["Application Orchestration Layer<br/>• EasyOCRLightningCLI (Custom CLI)<br/>• Main Entry Points (main.py, main_fittest.py)<br/>• Configuration Parser (JSONArgParse)"]
    
    TrainingWorkflow["Training<br/>Workflow"]
    TestingWorkflow["Testing<br/>Workflow"]
    PredictionWorkflow["Prediction<br/>Workflow"]
    
    LightningFramework["PyTorch Lightning Framework<br/>• Trainer (Training Loop Management)<br/>• LightningModule (Model Logic)<br/>• LightningDataModule (Data Management)<br/>• Callbacks (Checkpointing, Logging)<br/>• Loggers (TensorBoard)"]
    
    PyTorch["PyTorch<br/>Framework"]
    Utilities["Utilities<br/>(PIL, Pandas, jiwer, etc.)"]
    
    UILayer --> OrchestrationLayer
    OrchestrationLayer --> TrainingWorkflow
    OrchestrationLayer --> TestingWorkflow
    OrchestrationLayer --> PredictionWorkflow
    
    TrainingWorkflow --> LightningFramework
    TestingWorkflow --> LightningFramework
    PredictionWorkflow --> LightningFramework
    
    LightningFramework --> PyTorch
    LightningFramework --> Utilities
    
    style UILayer fill:#e3f2fd
    style OrchestrationLayer fill:#f3e5f5
    style LightningFramework fill:#fff9c4
    style PyTorch fill:#e8f5e9
    style Utilities fill:#e8f5e9
```

### Data Flow Architecture

```mermaid
flowchart TD
    RawData["Raw Images<br/>+ CSV Labels<br/>(labels.csv)"]
    
    OCRDataset["OCRDataset<br/>(OCR Text Recognition)"]
    EasyOCRDataModule["EasyOCRDataModule<br/>(Data Management)"]
    
    Preprocessing["Preprocessing<br/>• Load image (PIL)<br/>• Convert to grayscale/RGB<br/>• Resize with aspect ratio<br/>• Pad to fixed dimensions<br/>• Convert to tensor"]
    
    ModelModule["EasyOCRLitModule<br/>(Lightning Module)"]
    EasyOCRNet["EasyOCRNet<br/>(CRNN Model)"]
    
    PostProcessing["Post-Processing & Evaluation<br/>• CTC/Attention Decoding<br/>• WER/CER Metrics<br/>• Accuracy Computation<br/>• Generate CSV Results"]
    
    RawData --> OCRDataset
    OCRDataset --> EasyOCRDataModule
    EasyOCRDataModule --> Preprocessing
    Preprocessing --> ModelModule
    ModelModule --> EasyOCRNet
    EasyOCRNet --> PostProcessing
    
    style RawData fill:#e1f5ff
    style Preprocessing fill:#fff4e1
    style EasyOCRNet fill:#f3e5f5
    style PostProcessing fill:#e8f5e9
```

---

## Low-Level Design (LLD)

### Module Dependencies

```
EasyOCRTraining/
├── cli.py
│   └── depends on: lightning.pytorch.cli
│
├── main.py
│   └── depends on: cli.py
│
├── main_fittest.py
│   └── depends on: cli.py
│
├── datamodule.py
│   ├── depends on: lightning, EasyOCRTraining.dataset
│   └── exports: EasyOCRDataModule
│
├── dataset.py
│   ├── depends on: torch.utils.data, PIL, torchvision, pandas
│   └── exports: OCRDataset, AlignCollate
│
├── module.py
│   ├── depends on: lightning, torch, EasyOCRTraining.model, EasyOCRTraining.utilities.labelconverter, utils.easyocrmetrics
│   └── exports: EasyOCRLitModule
│
├── model.py
│   ├── depends on: torch.nn, EasyOCRTraining.utilities.*
│   └── exports: EasyOCRNet
│
└── utilities/
    ├── feature_extraction.py
    │   └── exports: VGG_FeatureExtractor, ResNet_FeatureExtractor, RCNN_FeatureExtractor, GRCL
    │
    ├── sequence_modeling.py
    │   └── exports: BidirectionalLSTM
    │
    ├── prediction.py
    │   └── exports: Attention, AttentionCell
    │
    ├── transformation.py
    │   └── exports: TPS_SpatialTransformerNetwork, LocalizationNetwork, GridGenerator
    │
    └── labelconverter.py
        └── exports: CTCLabelConverter, AttnLabelConverter

utils/ (EasyOCR-specific)
└── easyocrmetrics.py
    └── exports: EasyOCRMetricsManager
```

---

## Class Diagrams

### Core Classes Diagram

```mermaid
classDiagram
    class LightningModule {
        <<abstract>>
        +training_step()
        +validation_step()
        +test_step()
        +configure_optimizers()
    }
    
    class LightningDataModule {
        <<abstract>>
        +setup()
        +train_dataloader()
        +val_dataloader()
        +test_dataloader()
    }
    
    class LightningCLI {
        <<abstract>>
        +add_arguments_to_parser()
    }
    
    class Dataset {
        <<external>>
    }
    
    class EasyOCRLitModule {
        -EasyOCRNet model
        -CTCLabelConverter converter (CTC)
        -AttnLabelConverter converter (Attn)
        -nn.CTCLoss criterion (CTC)
        -nn.CrossEntropyLoss criterion (Attn)
        -EasyOCRMetricsManager metrics_manager
        -str Prediction
        -str character
        -int batch_max_length
        -str decode
        -str pretrained_model
        -str prediction_dir
        -List prediction_results
        +training_step(batch, batch_idx)
        +validation_step(batch, batch_idx)
        +on_validation_epoch_end()
        +test_step(batch, batch_idx)
        +predict_step(batch, batch_idx)
        +forward(images, text)
        -_runInference(batch)
        -_preparePrediction()
        -_writePredictionCSV(filename)
    }
    
    class EasyOCRDataModule {
        -str data_dir
        -int batch_size
        -int num_workers
        -int imgH
        -int imgW
        -bool keep_ratio
        -float contrast
        +setup(stage)
        +train_dataloader()
        +val_dataloader()
        +test_dataloader()
        +predict_dataloader()
        +transfer_batch_to_device(batch, device, dataloader_idx)
        -_buildDataset(data_root)
        -_buildDataLoader(dataset, shuffle, persistent_workers)
    }
    
    class OCRDataset {
        -str root_dir
        -str csv_file
        -str img_dir
        -bool grayscale
        -List filenames
        -List labels
        +__len__()
        +__getitem__(idx)
    }
    
    class AlignCollate {
        -int imgH
        -int imgW
        -bool keep_ratio
        +__call__(batch)
    }
    
    class EasyOCRLightningCLI {
        +add_arguments_to_parser(parser)
    }
    
    class EasyOCRNet {
        -dict stages
        -Optional TPS_SpatialTransformerNetwork Transformation
        -FeatureExtractor FeatureExtraction
        -nn.AdaptiveAvgPool2d AdaptiveAvgPool
        -Optional BidirectionalLSTM SequenceModeling
        -nn.Linear or Attention Prediction
        +forward(input, text, is_train)
    }
    
    class EasyOCRMetricsManager {
        -List all_ground_truths
        -List all_predictions
        +update(ground_truths, predictions)
        +compute() Dict
        +summarize() Dict
        +reset()
    }
    
    %% Inheritance
    LightningModule <|-- EasyOCRLitModule
    LightningDataModule <|-- EasyOCRDataModule
    LightningCLI <|-- EasyOCRLightningCLI
    Dataset <|-- OCRDataset
    
    %% Composition
    EasyOCRLitModule *-- EasyOCRNet : uses
    EasyOCRLitModule *-- EasyOCRMetricsManager : uses
    EasyOCRDataModule *-- OCRDataset : creates
    EasyOCRDataModule *-- AlignCollate : uses
    EasyOCRNet *-- FeatureExtractor : uses
    EasyOCRNet *-- BidirectionalLSTM : uses
    EasyOCRNet *-- Attention : uses
    EasyOCRNet *-- TPS_SpatialTransformerNetwork : uses
```

### Detailed EasyOCRLitModule Class

```mermaid
classDiagram
    class EasyOCRLitModule {
        -EasyOCRNet model
        -CTCLabelConverter or AttnLabelConverter converter
        -nn.CTCLoss or nn.CrossEntropyLoss criterion
        -EasyOCRMetricsManager metrics_manager
        -str Prediction
        -str character
        -int batch_max_length
        -str decode
        -str pretrained_model
        -bool freeze_FeatureExtraction
        -bool freeze_SequenceModeling
        -str prediction_dir
        -List prediction_results
        +__init__(character, pretrained_model, Prediction, batch_max_length, model, save_predicted_images, decode, freeze_FeatureExtraction, freeze_SequenceModeling)
        +forward(images, text) Tensor
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) Dict
        +on_validation_epoch_end()
        +test_step(batch, batch_idx)
        +predict_step(batch, batch_idx)
        -_runInference(batch)
        -_preparePrediction()
        -_writePredictionCSV(filename)
    }
```

### Detailed EasyOCRDataModule Class

```mermaid
classDiagram
    class EasyOCRDataModule {
        -str data_dir
        -int batch_size
        -int num_workers
        -int imgH
        -int imgW
        -bool keep_ratio
        -float contrast
        -str train_data_root
        -str valid_data_root
        -str test_data_root
        -str predict_data_root
        -OCRDataset train_dataset
        -OCRDataset valid_dataset
        -OCRDataset test_dataset
        -OCRDataset predict_dataset
        +__init__(data_dir, batch_size, num_workers, imgH, imgW, PAD, contrast_adjust)
        +setup(stage) str
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +test_dataloader() DataLoader
        +predict_dataloader() DataLoader
        +transfer_batch_to_device(batch, device, dataloader_idx)
        -_buildDataset(data_root) OCRDataset
        -_buildDataLoader(dataset, shuffle, persistent_workers) DataLoader
    }
```

### Detailed EasyOCRNet Class

```mermaid
classDiagram
    class EasyOCRNet {
        -dict stages
        -Optional TPS_SpatialTransformerNetwork Transformation
        -VGG_FeatureExtractor or ResNet_FeatureExtractor or RCNN_FeatureExtractor FeatureExtraction
        -int FeatureExtraction_output
        -nn.AdaptiveAvgPool2d AdaptiveAvgPool
        -Optional BidirectionalLSTM SequenceModeling
        -int SequenceModeling_output
        -nn.Linear or Attention Prediction
        +__init__(Transformation, FeatureExtraction, SequenceModeling, Prediction, input_channel, output_channel, hidden_size, num_fiducial, imgH, imgW, num_class)
        +forward(input, text, is_train) Tensor
    }
```

---

## EasyOCR Model Architecture

### Model Architecture Overview

```mermaid
flowchart TD
    Input["Input Image<br/>(B, 1, H, W)"]
    
    Transformation["Transformation Stage<br/>(Optional TPS)"]
    
    FeatureExtraction["Feature Extraction Stage<br/>(VGG/ResNet/RCNN)"]
    
    AdaptivePool["Adaptive AvgPool2d<br/>(None, 1)"]
    
    Permute["Permute<br/>(B, C, H, W) → (B, H, W, C)"]
    
    Squeeze["Squeeze<br/>(B, W, C)"]
    
    SequenceModeling["Sequence Modeling Stage<br/>(Optional BiLSTM)"]
    
    Prediction["Prediction Stage<br/>(CTC or Attention)"]
    
    Output["Output<br/>• CTC: (B, W, num_class)<br/>• Attention: (B, T, num_class)"]
    
    Input --> Transformation
    Transformation --> FeatureExtraction
    FeatureExtraction --> Permute
    Permute --> AdaptivePool
    AdaptivePool --> Squeeze
    Squeeze --> SequenceModeling
    SequenceModeling --> Prediction
    Prediction --> Output
    
    style Input fill:#e1f5ff
    style FeatureExtraction fill:#fff4e1
    style SequenceModeling fill:#f3e5f5
    style Prediction fill:#e8f5e9
```

### Four-Stage Architecture Detail

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Transformation<br/>(Optional)"]
        TPS["TPS Spatial Transformer<br/>Network"]
        None["None (Identity)"]
    end
    
    subgraph Stage2["Stage 2: Feature Extraction"]
        VGG["VGG Feature<br/>Extractor"]
        ResNet["ResNet Feature<br/>Extractor"]
        RCNN["RCNN Feature<br/>Extractor"]
    end
    
    subgraph Stage3["Stage 3: Sequence Modeling<br/>(Optional)"]
        BiLSTM["Bidirectional<br/>LSTM"]
        None2["None (Identity)"]
    end
    
    subgraph Stage4["Stage 4: Prediction"]
        CTC["CTC<br/>(Linear Layer)"]
        Attn["Attention<br/>(Seq2Seq)"]
    end
    
    Input --> Stage1
    Stage1 --> Stage2
    Stage2 --> Stage3
    Stage3 --> Stage4
    Stage4 --> Output
    
    style Stage1 fill:#e3f2fd
    style Stage2 fill:#fff3e0
    style Stage3 fill:#f3e5f5
    style Stage4 fill:#e8f5e9
```

### Feature Extraction Architectures

```mermaid
flowchart TD
    Input["Input<br/>(B, 1, H, W)"]
    
    subgraph VGG["VGG Feature Extractor"]
        V1["Conv(1→64) + ReLU + MaxPool"]
        V2["Conv(64→128) + ReLU + MaxPool"]
        V3["Conv(128→256) + ReLU"]
        V4["Conv(256→256) + ReLU + MaxPool(2,1)"]
        V5["Conv(256→512) + BN + ReLU"]
        V6["Conv(512→512) + BN + ReLU + MaxPool(2,1)"]
        V7["Conv(512→512) + ReLU"]
        VOut["Output (B, 512, H', W')"]
    end
    
    subgraph ResNet["ResNet Feature Extractor"]
        R1["Conv0_1 + Conv0_2"]
        R2["MaxPool1 + Layer1 + Conv1"]
        R3["MaxPool2 + Layer2 + Conv2"]
        R4["MaxPool3 + Layer3 + Conv3"]
        R5["Layer4 + Conv4_1 + Conv4_2"]
        ROut["Output (B, 512, H', W')"]
    end
    
    subgraph RCNN["RCNN Feature Extractor"]
        RC1["Conv + ReLU + MaxPool"]
        RC2["GRCL + MaxPool"]
        RC3["GRCL + MaxPool"]
        RC4["GRCL + MaxPool"]
        RC5["Conv + BN + ReLU"]
        RCOut["Output (B, 512, H', W')"]
    end
    
    Input --> VGG
    Input --> ResNet
    Input --> RCNN
    
    V1 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7 --> VOut
    R1 --> R2 --> R3 --> R4 --> R5 --> ROut
    RC1 --> RC2 --> RC3 --> RC4 --> RC5 --> RCOut
    
    style VGG fill:#fff4e1
    style ResNet fill:#f3e5f5
    style RCNN fill:#e8f5e9
```

### Sequence Modeling Flow

```mermaid
flowchart LR
    VisualFeature["Visual Features<br/>(B, W, C)"]
    
    BiLSTM1["BiLSTM Layer 1<br/>(C → hidden_size)"]
    
    BiLSTM2["BiLSTM Layer 2<br/>(hidden_size → hidden_size)"]
    
    ContextualFeature["Contextual Features<br/>(B, W, hidden_size)"]
    
    VisualFeature --> BiLSTM1
    BiLSTM1 --> BiLSTM2
    BiLSTM2 --> ContextualFeature
    
    style VisualFeature fill:#e1f5ff
    style ContextualFeature fill:#e8f5e9
```

### Prediction Stage: CTC Flow

```mermaid
flowchart TD
    ContextualFeature["Contextual Features<br/>(B, W, hidden_size)"]
    
    Linear["Linear Layer<br/>(hidden_size → num_class)"]
    
    Logits["Logits<br/>(B, W, num_class)"]
    
    LogSoftmax["Log Softmax<br/>(B, W, num_class)"]
    
    Permute["Permute<br/>(W, B, num_class)"]
    
    CTC["CTC Loss/Decoding<br/>• Remove blanks<br/>• Remove repeats"]
    
    Output["Text Sequence"]
    
    ContextualFeature --> Linear
    Linear --> Logits
    Logits --> LogSoftmax
    LogSoftmax --> Permute
    Permute --> CTC
    CTC --> Output
    
    style ContextualFeature fill:#e1f5ff
    style CTC fill:#fff4e1
    style Output fill:#e8f5e9
```

### Prediction Stage: Attention Flow

```mermaid
flowchart TD
    ContextualFeature["Contextual Features<br/>(B, W, hidden_size)"]
    
    AttentionCell["Attention Cell<br/>• Compute attention weights<br/>• Generate context<br/>• LSTM step"]
    
    Generator["Generator<br/>(Linear: hidden_size → num_class)"]
    
    Probs["Probabilities<br/>(B, T, num_class)"]
    
    subgraph Training["Training Mode<br/>(Teacher Forcing)"]
        TextInput["Text Input<br/>(Ground Truth)"]
        TextInput --> AttentionCell
    end
    
    subgraph Inference["Inference Mode<br/>(Autoregressive)"]
        PrevPred["Previous Prediction"]
        PrevPred --> AttentionCell
    end
    
    ContextualFeature --> AttentionCell
    AttentionCell --> Generator
    Generator --> Probs
    
    style ContextualFeature fill:#e1f5ff
    style AttentionCell fill:#fff4e1
    style Probs fill:#e8f5e9
```

### TPS Transformation Flow

```mermaid
flowchart TD
    InputImage["Input Image<br/>(B, C, H, W)"]
    
    LocalizationNet["Localization Network<br/>• Conv layers<br/>• FC layers<br/>• Predict fiducials"]
    
    Fiducials["Fiducial Points<br/>(B, F, 2)"]
    
    GridGenerator["Grid Generator<br/>• Build TPS matrix<br/>• Generate grid"]
    
    Grid["Transformation Grid<br/>(B, H, W, 2)"]
    
    GridSample["Grid Sample<br/>(Bilinear Sampling)"]
    
    RectifiedImage["Rectified Image<br/>(B, C, H, W)"]
    
    InputImage --> LocalizationNet
    LocalizationNet --> Fiducials
    Fiducials --> GridGenerator
    InputImage --> GridGenerator
    GridGenerator --> Grid
    InputImage --> GridSample
    Grid --> GridSample
    GridSample --> RectifiedImage
    
    style InputImage fill:#e1f5ff
    style GridGenerator fill:#fff4e1
    style RectifiedImage fill:#e8f5e9
```

---

## Sequence Diagrams

### Training Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ODL as EasyOCRLightningCLI
    participant Trainer
    participant Model as EasyOCRLitModule
    participant Data as EasyOCRDataModule
    participant Dataset as OCRDataset
    participant EasyOCRNet

    User->>CLI: python main.py fit --config ...
    CLI->>ODL: EasyOCRLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>Data: EasyOCRDataModule(...)
    ODL->>Model: EasyOCRLitModule(...)
    ODL->>Model: EasyOCRNet(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: fit(model, datamodule)

    Trainer->>Data: setup("fit")
    Data->>Dataset: OCRDataset(...)
    Data->>Dataset: setup("fit")
    Note over Dataset: Load images from images/<br/>Load labels from labels.csv
    Dataset-->>Data: train_dataset, valid_dataset

    Trainer->>Model: Load pretrained weights<br/>(if specified)

    loop For each epoch
        Trainer->>Data: train_dataloader()
        Data->>Dataset: DataLoader(train_dataset, AlignCollate)
        Data-->>Trainer: train_dataloader
        
        loop For each batch
            Trainer->>Data: Get batch
            Data->>Dataset: __getitem__(idx)
            Dataset-->>Data: (image, label)
            Data->>Data: AlignCollate() (resize & pad)
            Data-->>Trainer: batch (images, labels)
            
            Trainer->>Model: training_step(batch, batch_idx)
            Model->>Model: converter.encode(labels)
            Model->>EasyOCRNet: forward(images, text_input)
            
            alt CTC Prediction
                EasyOCRNet->>EasyOCRNet: Transformation (if TPS)
                EasyOCRNet->>EasyOCRNet: FeatureExtraction
                EasyOCRNet->>EasyOCRNet: AdaptiveAvgPool + Squeeze
                EasyOCRNet->>EasyOCRNet: SequenceModeling (if BiLSTM)
                EasyOCRNet->>EasyOCRNet: Linear Prediction
                EasyOCRNet-->>Model: logits (B, W, num_class)
                Model->>Model: log_softmax + permute
                Model->>Model: CTCLoss
            else Attention Prediction
                EasyOCRNet->>EasyOCRNet: Transformation (if TPS)
                EasyOCRNet->>EasyOCRNet: FeatureExtraction
                EasyOCRNet->>EasyOCRNet: AdaptiveAvgPool + Squeeze
                EasyOCRNet->>EasyOCRNet: SequenceModeling (if BiLSTM)
                EasyOCRNet->>EasyOCRNet: Attention Prediction
                EasyOCRNet-->>Model: probs (B, T, num_class)
                Model->>Model: CrossEntropyLoss
            end
            
            Model-->>Trainer: loss
            Trainer->>Trainer: backpropagation
            Trainer->>Trainer: optimizer.step()
        end
        
        Trainer->>Data: val_dataloader()
        Data-->>Trainer: val_dataloader
        
        loop For each validation batch
            Trainer->>Data: Get batch
            Data-->>Trainer: batch
            Trainer->>Model: validation_step(batch, batch_idx)
            Model->>Model: Forward pass + decode
            Model->>Model: Compute accuracy
            Model->>Model: metrics_manager.update()
            Model-->>Trainer: val_loss, val_acc
        end
        
        Trainer->>Model: on_validation_epoch_end()
        Model->>Model: metrics_manager.compute()
        Model->>Model: Log WER, CER
        Model-->>Trainer: Validation complete
        
        Trainer->>Trainer: Log metrics
        Trainer->>Trainer: Save checkpoint
    end

    Trainer-->>CLI: Training complete
    CLI-->>User: Done
```

### Testing Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ODL as EasyOCRLightningCLI
    participant Trainer
    participant Model as EasyOCRLitModule
    participant Data as EasyOCRDataModule
    participant Dataset as OCRDataset
    participant EasyOCRNet

    User->>CLI: python main.py test --config ... --ckpt_path ...
    CLI->>ODL: EasyOCRLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>ODL: load_checkpoint()
    ODL->>Data: EasyOCRDataModule(...)
    ODL->>Model: EasyOCRLitModule.load_from_checkpoint(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: test(model, datamodule)

    Trainer->>Data: setup("test")
    Data->>Dataset: OCRDataset(...)
    Data-->>Trainer: test_dataset

    Trainer->>Model: on_test_epoch_start()
    Model->>Model: _preparePrediction()
    Note over Model: Create prediction_dir

    Trainer->>Data: test_dataloader()
    Data-->>Trainer: test_dataloader

    loop For each batch
        Trainer->>Data: Get batch
        Data->>Dataset: __getitem__(idx)
        Dataset-->>Data: (image, label)
        Data-->>Trainer: batch
        
        Trainer->>Model: test_step(batch, batch_idx)
        Model->>Model: _runInference(batch)
        Model->>EasyOCRNet: forward(images, None)
        EasyOCRNet-->>Model: logits or probs
        
        alt CTC Prediction
            Model->>Model: log_softmax + permute
            Model->>Model: argmax
            Model->>Model: converter.decode_greedy() or decode_beamsearch()
        else Attention Prediction
            Model->>Model: softmax + argmax
            Model->>Model: converter.decode()
        end
        
        Model->>Model: Store predictions
    end

    Trainer->>Model: on_test_epoch_end()
    Model->>Model: _writePredictionCSV("output.csv")
    Model-->>Trainer: Test complete

    Trainer-->>CLI: Testing complete
    CLI-->>User: Done
```

### Prediction Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ODL as EasyOCRLightningCLI
    participant Trainer
    participant Model as EasyOCRLitModule
    participant Data as EasyOCRDataModule
    participant Dataset as OCRDataset
    participant EasyOCRNet

    User->>CLI: python main.py predict --config ... --data.data_dir ...
    CLI->>ODL: EasyOCRLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>ODL: load_checkpoint()
    ODL->>Data: EasyOCRDataModule(...)
    ODL->>Model: EasyOCRLitModule.load_from_checkpoint(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: predict(model, datamodule)

    Trainer->>Data: setup("predict")
    Data->>Dataset: OCRDataset(...)
    Data-->>Trainer: predict_dataset

    Trainer->>Model: on_predict_epoch_start()
    Model->>Model: _preparePrediction()
    Note over Model: Create prediction_dir

    Trainer->>Data: predict_dataloader()
    Data-->>Trainer: predict_dataloader

    loop For each batch
        Trainer->>Data: Get batch
        Data->>Dataset: __getitem__(idx)
        Dataset-->>Data: (image, label)
        Data-->>Trainer: batch
        
        Trainer->>Model: predict_step(batch, batch_idx)
        Model->>Model: _runInference(batch)
        Model->>EasyOCRNet: forward(images, None)
        EasyOCRNet-->>Model: logits or probs
        Model->>Model: Decode predictions
        Model->>Model: Store predictions
    end

    Trainer->>Model: on_predict_epoch_end()
    Model->>Model: _writePredictionCSV("predict_epoch{N}.csv")
    Model-->>Trainer: Prediction complete

    Trainer-->>CLI: Done
    CLI-->>User: Results saved in prediction/
```

---

## Folder and File Structure

### Complete Project Structure

```mermaid
graph TD
    Root["EasyOCRTraining/"]
    
    Config["config/"]
    ConfigYAML["easyOCR.yaml"]
    
    Utilities["utilities/"]
    FeatureExtract["feature_extraction.py<br/>VGG, ResNet, RCNN"]
    SeqModeling["sequence_modeling.py<br/>BidirectionalLSTM"]
    Prediction["prediction.py<br/>Attention"]
    Transform["transformation.py<br/>TPS"]
    LabelConv["labelconverter.py<br/>CTC, Attn"]
    
    CLI["cli.py<br/>EasyOCRLightningCLI"]
    Main["main.py<br/>CLI Entry Point"]
    MainFitTest["main_fittest.py<br/>Fit + Test"]
    DataModule["datamodule.py<br/>EasyOCRDataModule"]
    Dataset["dataset.py<br/>OCRDataset, AlignCollate"]
    Module["module.py<br/>EasyOCRLitModule"]
    Model["model.py<br/>EasyOCRNet"]
    
    Utils["utils/"]
    Metrics["easyocrmetrics.py<br/>EasyOCRMetricsManager"]
    
    Notebook["EasyOCRTrainer.ipynb"]
    Readme["README.md"]
    UML["UML_DIAGRAMS.md"]
    
    Root --> Config
    Root --> Utilities
    Root --> CLI
    Root --> Main
    Root --> MainFitTest
    Root --> DataModule
    Root --> Dataset
    Root --> Module
    Root --> Model
    Root --> Readme
    Root --> UML
    
    Config --> ConfigYAML
    
    Utilities --> FeatureExtract
    Utilities --> SeqModeling
    Utilities --> Prediction
    Utilities --> Transform
    Utilities --> LabelConv
    
    Utils --> Metrics
    
    style Root fill:#e1f5ff
    style Model fill:#fff4e1
    style Module fill:#e8f5e9
    style Utilities fill:#f3e5f5
```

### Dependency Graph

```mermaid
flowchart TD
    ExtDeps["External Dependencies<br/>• PyTorch (torch)<br/>• PyTorch Lightning (lightning)<br/>• PIL (Pillow)<br/>• Pandas<br/>• JSONArgParse<br/>• TensorBoard<br/>• jiwer"]
    
    EasyOCRTraining["EasyOCRTraining"]
    Utils["utils/ (EasyOCR)"]
    
    Application["OCR Text Recognition<br/>Application"]
    
    ExtDeps -->|used by| EasyOCRTraining
    ExtDeps -->|used by| Utils
    
    EasyOCRTraining -->|uses| Application
    Utils -->|uses| Application
    
    style ExtDeps fill:#e3f2fd
    style Application fill:#e8f5e9
```

---

## Module Interconnections

### Import Dependencies

```mermaid
graph LR
    subgraph EasyOCRTraining["EasyOCRTraining/"]
        CLI["cli.py"]
        Main["main.py"]
        MainFitTest["main_fittest.py"]
        DataModule["datamodule.py"]
        Dataset["dataset.py"]
        Module["module.py"]
        Model["model.py"]
    end
    
    subgraph Utilities["utilities/"]
        FeatureExtract["feature_extraction.py"]
        SeqModeling["sequence_modeling.py"]
        Prediction["prediction.py"]
        Transform["transformation.py"]
        LabelConv["labelconverter.py"]
    end
    
    subgraph Utils["utils/"]
        Metrics["easyocrmetrics.py"]
    end
    
    subgraph External["External Libraries"]
        Lightning["lightning.pytorch.cli"]
        Torch["torch"]
        PIL["PIL"]
        Pandas["pandas"]
        JSONArgParse["jsonargparse"]
        TensorBoard["tensorboard"]
        Jiwer["jiwer"]
    end
    
    CLI --> Lightning
    Main --> CLI
    MainFitTest --> CLI
    DataModule --> Lightning
    DataModule --> Dataset
    Dataset --> Torch
    Dataset --> PIL
    Dataset --> Pandas
    Module --> Lightning
    Module --> Torch
    Module --> Model
    Module --> LabelConv
    Module --> Metrics
    Model --> Torch
    Model --> FeatureExtract
    Model --> SeqModeling
    Model --> Prediction
    Model --> Transform
    
    FeatureExtract --> Torch
    SeqModeling --> Torch
    Prediction --> Torch
    Transform --> Torch
    LabelConv --> Torch
    Metrics --> Jiwer
    
    style EasyOCRTraining fill:#fff4e1
    style Utilities fill:#f3e5f5
    style External fill:#e3f2fd
```

### Data Flow Between Modules

```mermaid
flowchart TD
    ConfigLayer["Configuration Layer<br/>(easyOCR.yaml)"]
    
    CLILayer["CLI Layer<br/>(EasyOCRLightningCLI, main.py,<br/>main_fittest.py)"]
    
    DataModule["EasyOCRDataModule"]
    ModelModule["EasyOCRLitModule"]
    Trainer["Trainer"]
    
    OCRDataset["OCRDataset"]
    EasyOCRNet["EasyOCRNet"]
    
    UtilsModules["Utility Modules<br/>(feature_extraction,<br/>sequence_modeling,<br/>prediction,<br/>transformation,<br/>labelconverter)"]
    
    Execution["Training/Test/<br/>Prediction<br/>Execution"]
    
    ConfigLayer -->|config| CLILayer
    CLILayer -->|creates| DataModule
    CLILayer -->|creates| ModelModule
    CLILayer -->|creates| Trainer
    
    DataModule -->|uses| OCRDataset
    ModelModule -->|uses| EasyOCRNet
    Trainer -->|manages| DataModule
    Trainer -->|manages| ModelModule
    
    EasyOCRNet -->|uses| UtilsModules
    
    DataModule --> Execution
    ModelModule --> Execution
    Trainer --> Execution
    
    style ConfigLayer fill:#e3f2fd
    style CLILayer fill:#f3e5f5
    style Execution fill:#e8f5e9
```

---

## Data Flow Diagrams

### Training Data Flow

```mermaid
flowchart TD
    ImageFiles["Image Files<br/>(images/)"]
    CSVFile["CSV File<br/>(labels.csv)"]
    
    OCRDataset["OCRDataset.__getitem__()<br/>• Load image (PIL)<br/>• Load label from CSV<br/>• Convert to grayscale/RGB<br/>• Return (image, label)"]
    
    AlignCollate["AlignCollate()<br/>• Resize with aspect ratio<br/>• Pad to fixed dimensions<br/>• Convert to tensor<br/>• Stack into batch"]
    
    Encode["Label Encoding<br/>• CTC: text → indices (with blank)<br/>• Attention: text → indices (with [GO], [s])"]
    
    ModelForward["Model Forward Pass<br/>• Transformation (optional TPS)<br/>• Feature Extraction (VGG/ResNet/RCNN)<br/>• AdaptiveAvgPool + Squeeze<br/>• Sequence Modeling (optional BiLSTM)<br/>• Prediction (CTC or Attention)"]
    
    Loss["Loss Computation<br/>• CTC: CTCLoss<br/>• Attention: CrossEntropyLoss"]
    
    TrainingStep["training_step()<br/>• Backward pass<br/>• Optimizer step"]
    
    ImageFiles --> OCRDataset
    CSVFile --> OCRDataset
    OCRDataset --> AlignCollate
    AlignCollate --> Encode
    Encode --> ModelForward
    ModelForward --> Loss
    Loss --> TrainingStep
    
    style ImageFiles fill:#e1f5ff
    style CSVFile fill:#e1f5ff
    style TrainingStep fill:#e8f5e9
```

### Inference Data Flow

```mermaid
flowchart TD
    ImageFiles["Image Files<br/>(images/)"]
    
    OCRDataset["OCRDataset.__getitem__()<br/>• Load image<br/>• Load label (optional)<br/>• Convert to grayscale/RGB"]
    
    AlignCollate["AlignCollate()<br/>• Resize with aspect ratio<br/>• Pad to fixed dimensions<br/>• Convert to tensor"]
    
    ModelForward["Model Forward Pass<br/>• Transformation (if TPS)<br/>• Feature Extraction<br/>• AdaptiveAvgPool + Squeeze<br/>• Sequence Modeling (if BiLSTM)<br/>• Prediction"]
    
    Decode["Decoding<br/>• CTC: decode_greedy() or decode_beamsearch()<br/>• Attention: decode()"]
    
    PostProcess["Post-Processing<br/>• Remove blanks (CTC)<br/>• Remove repeats (CTC)<br/>• Remove tokens (Attention)"]
    
    SaveResults["Save Results<br/>• CSV file (ground_truth, predicted_text)<br/>• Update metrics (test only)"]
    
    ImageFiles --> OCRDataset
    OCRDataset --> AlignCollate
    AlignCollate --> ModelForward
    ModelForward --> Decode
    Decode --> PostProcess
    PostProcess --> SaveResults
    
    style ImageFiles fill:#e1f5ff
    style ModelForward fill:#fff4e1
    style Decode fill:#f3e5f5
    style SaveResults fill:#e8f5e9
```

### CTC Encoding/Decoding Flow

```mermaid
flowchart TD
    Text["Text String<br/>e.g., 'hello'"]
    
    Encode["CTC Encoding<br/>• Map chars to indices (1-N)<br/>• 0 reserved for blank<br/>• Example: 'hello' → [8,5,12,12,15]"]
    
    ModelOutput["Model Output<br/>• Logits (B, W, num_class)<br/>• Log softmax<br/>• Permute (W, B, num_class)"]
    
    Argmax["Argmax<br/>• Get indices<br/>• Example: [8,0,5,12,12,0,15]"]
    
    Decode["CTC Decoding<br/>• Remove blanks (0)<br/>• Remove repeats<br/>• Example: [8,0,5,12,12,0,15] → 'hello'"]
    
    Output["Decoded Text<br/>'hello'"]
    
    Text --> Encode
    Encode --> ModelOutput
    ModelOutput --> Argmax
    Argmax --> Decode
    Decode --> Output
    
    style Text fill:#e1f5ff
    style Encode fill:#fff4e1
    style Decode fill:#f3e5f5
    style Output fill:#e8f5e9
```

### Attention Encoding/Decoding Flow

```mermaid
flowchart TD
    Text["Text String<br/>e.g., 'hello'"]
    
    Encode["Attention Encoding<br/>• Add [GO] at start<br/>• Add [s] at end<br/>• Map to indices<br/>• Example: 'hello' → [[GO],'h','e','l','l','o',[s]]"]
    
    ModelForward["Model Forward<br/>• Teacher forcing (training)<br/>• Autoregressive (inference)"]
    
    ModelOutput["Model Output<br/>• Probabilities (B, T, num_class)<br/>• Softmax<br/>• Argmax"]
    
    Decode["Attention Decoding<br/>• Map indices to chars<br/>• Remove tokens<br/>• Example: [[GO],'h','e','l','l','o',[s]] → 'hello'"]
    
    Output["Decoded Text<br/>'hello'"]
    
    Text --> Encode
    Encode --> ModelForward
    ModelForward --> ModelOutput
    ModelOutput --> Decode
    Decode --> Output
    
    style Text fill:#e1f5ff
    style Encode fill:#fff4e1
    style Decode fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

## Configuration Flow

```mermaid
flowchart TD
    BaseConfig["easyOCR.yaml<br/>(Base Configuration)"]
    
    Parser["JSONArgParse / LightningCLI"]
    
    TrainerConfig["Trainer<br/>Config"]
    ModelConfig["Model<br/>Config<br/>(EasyOCRLitModule + EasyOCRNet)"]
    DataConfig["Data<br/>Config<br/>(EasyOCRDataModule)"]
    
    Execution["PyTorch Lightning<br/>Execution"]
    
    BaseConfig -->|parsed by| Parser
    Parser -->|creates| TrainerConfig
    Parser -->|creates| ModelConfig
    Parser -->|creates| DataConfig
    TrainerConfig -->|instantiates| Execution
    ModelConfig -->|instantiates| Execution
    DataConfig -->|instantiates| Execution
    
    style BaseConfig fill:#e3f2fd
    style Execution fill:#e8f5e9
```

---

## Summary

This document provides comprehensive diagrams and design documentation for the EasyOCRTraining project. The system follows a modular, extensible architecture based on PyTorch Lightning, making it easy to:

1. **Train OCR models** for text recognition tasks
2. **Customize model architecture** via configuration (feature extractor, sequence modeling, prediction method)
3. **Extend functionality** through utility modules
4. **Configure workflows** through YAML files

The design separates concerns:
- **EasyOCRTraining/**: Core OCR training components
- **utilities/**: Model component implementations (feature extraction, sequence modeling, prediction, transformation, label conversion)
- **utils/**: Shared utilities (metrics)

The EasyOCR model uses a four-stage architecture (Transformation → Feature Extraction → Sequence Modeling → Prediction) with flexible components for each stage, supporting multiple feature extractors (VGG, ResNet, RCNN) and prediction methods (CTC, Attention) to handle various text recognition scenarios.
