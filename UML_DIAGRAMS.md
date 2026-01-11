# UML Diagrams and Design Documentation

This document contains comprehensive UML diagrams, class diagrams, sequence diagrams, High-Level Design (HLD), Low-Level Design (LLD), and visual representations of the project structure and interconnections.

## Table of Contents

- [High-Level Design (HLD)](#high-level-design-hld)
- [Low-Level Design (LLD)](#low-level-design-lld)
- [Class Diagrams](#class-diagrams)
- [Sequence Diagrams](#sequence-diagrams)
- [Folder and File Structure](#folder-and-file-structure)
- [Module Interconnections](#module-interconnections)
- [Data Flow Diagrams](#data-flow-diagrams)

---

## High-Level Design (HLD)

### System Overview

```mermaid
flowchart TD
    System["Image-to-Physical System<br/>Object Detection Framework"]
    
    Config["Configuration<br/>Management"]
    Training["Training<br/>Pipeline"]
    Inference["Inference<br/>Pipeline"]
    
    PyTorchLightning["PyTorch<br/>Lightning<br/>Framework"]
    ModelEngine["Model<br/>Inference<br/>Engine"]
    
    DataModule["Data<br/>Module"]
    ModelModule["Model<br/>Module"]
    UtilsModule["Utils<br/>Modules"]
    
    DatasetVR["DatasetVR<br/>Dataset"]
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
    
    DataModule --> DatasetVR
    ModelModule --> DatasetVR
    UtilsModule --> DatasetVR
    
    ConfigFiles --> DatasetVR
    Notebook --> DatasetVR
    
    style System fill:#e1f5ff
    style DataModule fill:#fff4e1
    style ModelModule fill:#fff4e1
    style UtilsModule fill:#fff4e1
    style DatasetVR fill:#e8f5e9
```

### Component Architecture

```mermaid
flowchart TD
    UILayer["User Interface Layer<br/>• CLI (Command Line Interface)<br/>• Jupyter Notebook<br/>• Configuration Files (YAML)"]
    
    OrchestrationLayer["Application Orchestration Layer<br/>• ODLightningCLI (Custom CLI)<br/>• Main Entry Points (main.py, main_fittest.py)<br/>• Configuration Parser (JSONArgParse)"]
    
    TrainingWorkflow["Training<br/>Workflow"]
    TestingWorkflow["Testing<br/>Workflow"]
    PredictionWorkflow["Prediction<br/>Workflow"]
    
    LightningFramework["PyTorch Lightning Framework<br/>• Trainer (Training Loop Management)<br/>• LightningModule (Model Logic)<br/>• LightningDataModule (Data Management)<br/>• Callbacks (Checkpointing, Logging)<br/>• Loggers (TensorBoard)"]
    
    PyTorch["PyTorch<br/>Framework"]
    TorchVision["TorchVision<br/>Models<br/>(FasterRCNN)"]
    Utilities["Utilities<br/>(OpenCV, NumPy, etc.)"]
    
    UILayer --> OrchestrationLayer
    OrchestrationLayer --> TrainingWorkflow
    OrchestrationLayer --> TestingWorkflow
    OrchestrationLayer --> PredictionWorkflow
    
    TrainingWorkflow --> LightningFramework
    TestingWorkflow --> LightningFramework
    PredictionWorkflow --> LightningFramework
    
    LightningFramework --> PyTorch
    LightningFramework --> TorchVision
    LightningFramework --> Utilities
    
    style UILayer fill:#e3f2fd
    style OrchestrationLayer fill:#f3e5f5
    style LightningFramework fill:#fff9c4
    style PyTorch fill:#e8f5e9
    style TorchVision fill:#e8f5e9
    style Utilities fill:#e8f5e9
```

### Data Flow Architecture

```mermaid
flowchart TD
    RawData["Raw Images<br/>+ CSV Annotations"]
    
    DatasetVR["DatasetVR<br/>(View Recognition)"]
    DataModuleOD["DataModuleOD<br/>(Data Management)"]
    
    Preprocessing["Preprocessing & Augmentation<br/>• Image Preprocessing (Grayscale, Binary)<br/>• Data Augmentation (Training only)<br/>• Resize & Normalize"]
    
    ModelModuleOD["ModelModuleOD<br/>(Lightning Module)"]
    FasterRCNN["Faster R-CNN<br/>+ ResNet50 FPN"]
    
    PostProcessing["Post-Processing & Evaluation<br/>• Filter by Detection Threshold<br/>• Compute Metrics (mAP, Confusion Matrix)<br/>• Generate Visualizations<br/>• Export Results (CSV, Images)"]
    
    RawData --> DatasetVR
    DatasetVR --> DataModuleOD
    DataModuleOD --> Preprocessing
    Preprocessing --> ModelModuleOD
    ModelModuleOD --> FasterRCNN
    FasterRCNN --> PostProcessing
    
    style RawData fill:#e1f5ff
    style Preprocessing fill:#fff4e1
    style FasterRCNN fill:#f3e5f5
    style PostProcessing fill:#e8f5e9
```

---

## Low-Level Design (LLD)

### Module Dependencies

```
ObjectDetection/
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
│   ├── depends on: lightning, utils.lib, ObjectDetection.dataset, utils.dataset, utils.imageaugmentation
│   └── exports: DataModuleOD
│
├── dataset.py
│   ├── depends on: torch.utils.data, cv2, numpy, utils.imageaugmentation
│   └── exports: DatasetOD, DatasetImage
│
├── modelmodule.py
│   ├── depends on: lightning, utils.lib, utils.imagevisualization, utils.colors, torchmetrics, utils.metrics
│   └── exports: ModelModuleOD
│
└── modelfactory.py
    └── depends on: torch, torchvision
    └── exports: getModelfasterrcnn_resnet50_fpn

ViewRecognition/
├── dataset.py
│   ├── depends on: ObjectDetection.dataset, utils.imageaugmentation, utils.image
│   └── exports: DatasetVR
│
└── config/
    ├── viewrecognition.yaml
    └── viewrecognition.local.yaml

utils/
├── colors.py
│   └── exports: getRandomBASEColors, getRandomTABLEAUColors, hex_to_bgr
│
├── dataset.py
│   └── exports: collate_fn
│
├── image.py
│   └── exports: convertToBoundingBox
│
├── imageaugmentation.py
│   └── exports: preProcess, getTransform, getNoTransform
│
├── imagevisualization.py
│   └── exports: drawCV2BBWithText, visualizeOneBatchImages, visualizeImage
│
├── lib.py
│   └── exports: getCallableAndArgs, getAttr
│
└── metrics.py
    └── exports: computeIOU, computeBBConfusionMatrix
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
    
    class ModelModuleOD {
        -FasterRCNN model
        -int num_classes
        -float detection_threshold
        -MeanAveragePrecision metric_map
        -List~List~int~~ confusion_matrix
        +training_step(batch, batch_idx)
        +validation_step(batch, batch_idx)
        +test_step(batch, batch_idx)
        +predict_step(batch, batch_idx)
        -_runModel(batch, batch_idx)
        -_runInference(batch, batch_idx, computeMetrics)
        -_preparePrediction()
    }
    
    class DataModuleOD {
        -str data_dir
        -int batch_size
        -int num_workers
        -int resize
        -dict classes
        -type dataset_class
        -dict dataset_args
        +setup(stage)
        +train_dataloader()
        +val_dataloader()
        +test_dataloader()
        +predict_dataloader()
        -_buildDataset(data_dir, transforms)
        -_buildDataLoader(dataset, persistent_workers, shuffle)
    }
    
    class DatasetOD {
        <<abstract>>
        #str data_dir
        #int resize
        #dict classes
        #Any transforms
        #List~str~ all_images
        +__len__()
        +_getImageDetails(idx)
        +__getitem__(idx)*
    }
    
    class DatasetImage {
        +__getitem__(idx)
    }
    
    class DatasetVR {
        +__getitem__(idx)
        -_loadAnnotations(image_output_path)
        -_convertToBoundingBoxes(df, image_width, image_height)
        -_applyTransforms(image, target)
    }
    
    class ODLightningCLI {
        +add_arguments_to_parser(parser)
    }
    
    class imageaugmentation {
        <<module>>
        +preProcess(image, resize)
        +getTransform()
        +getNoTransform()
    }
    
    class imagevisualization {
        <<module>>
        +drawCV2BBWithText(image, bb, text, color)
        +visualizeOneBatchImages(batch)
        +visualizeImage(image, target, classes)
    }
    
    class metrics {
        <<module>>
        +computeIOU(box1, box2)
        +computeBBConfusionMatrix(target, output, num_classes)
    }
    
    class colors {
        <<module>>
        +getRandomBASEColors(count)
        +getRandomTABLEAUColors(count)
        +hex_to_bgr(hex_color)
    }
    
    class lib {
        <<module>>
        +getCallableAndArgs(class_dict, function_dict)
        +getAttr(path)
    }
    
    %% Inheritance
    LightningModule <|-- ModelModuleOD
    LightningDataModule <|-- DataModuleOD
    LightningCLI <|-- ODLightningCLI
    Dataset <|-- DatasetOD
    DatasetOD <|-- DatasetImage
    DatasetOD <|-- DatasetVR
    
    %% Composition
    ModelModuleOD *-- DataModuleOD : uses
    DataModuleOD *-- DatasetOD : creates
    
    %% Dependencies
    ModelModuleOD ..> imagevisualization : uses
    ModelModuleOD ..> metrics : uses
    ModelModuleOD ..> colors : uses
    ModelModuleOD ..> lib : uses
    DataModuleOD ..> imageaugmentation : uses
    DataModuleOD ..> lib : uses
    DatasetVR ..> imageaugmentation : uses
```

### Detailed ModelModuleOD Class

```mermaid
classDiagram
    class ModelModuleOD {
        -FasterRCNN model
        -int num_classes
        -float detection_threshold
        -bool visualize_training_images
        -bool save_predicted_images
        -List~List~int~~ confusion_matrix
        -MeanAveragePrecision metric_map
        -str prediction_dir
        -dict classcolors
        +__init__(num_classes, detection_threshold, visualize_training_images, save_predicted_images, torch_model, torch_model_factory)
        +training_step(batch, batch_idx) losses
        +validation_step(batch, batch_idx)
        +test_step(batch, batch_idx)
        +predict_step(batch, batch_idx)
        +on_train_batch_start(batch, batch_idx)
        +on_test_epoch_start()
        +on_test_epoch_end()
        +on_predict_epoch_start()
        -_runModel(batch, batch_idx) Tuple~losses, loss_value~
        -_runInference(batch, batch_idx, computeMetrics)
        -_preparePrediction()
    }
```

### Detailed DataModuleOD Class

```mermaid
classDiagram
    class DataModuleOD {
        -str data_dir
        -int batch_size
        -int num_workers
        -int resize
        -dict classes
        -type dataset_class
        -dict dataset_args
        -DatasetOD dataset_train
        -DatasetOD dataset_validate
        -DatasetOD dataset_test
        -DatasetImage dataset_predict
        +__init__(data_dir, batch_size, num_workers, resize, classes, dataset, dataset_factory)
        +setup(stage) str
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +test_dataloader() DataLoader
        +predict_dataloader() DataLoader
        +transfer_batch_to_device(batch, device, dataloader_idx)
        -_buildDataset(data_dir, transforms) DatasetOD
        -_buildDataLoader(dataset, persistent_workers, shuffle) DataLoader
    }
```

### Dataset Class Hierarchy

```mermaid
classDiagram
    class Dataset {
        <<external>>
        +__len__()
        +__getitem__(idx)
    }
    
    class DatasetOD {
        <<abstract>>
        #str data_dir
        #int resize
        #dict classes
        #Any transforms
        #List~str~ all_images
        +__len__() int
        +_getImageDetails(idx) tuple
        +__getitem__(idx)*
    }
    
    class DatasetVR {
        +__getitem__(idx)
        -Load CSV annotations
        -Filter labels
        -Convert to bboxes
        -Apply transforms
    }
    
    class DatasetImage {
        +__getitem__(idx)
        -Load image
        -Preprocess
        -No targets
    }
    
    Dataset <|-- DatasetOD
    DatasetOD <|-- DatasetVR
    DatasetOD <|-- DatasetImage
```

---

## Sequence Diagrams

### Training Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ODL as ODLightningCLI
    participant Trainer
    participant Model as ModelModuleOD
    participant Data as DataModuleOD
    participant Dataset as DatasetVR
    participant FasterRCNN

    User->>CLI: python main.py fit --config ...
    CLI->>ODL: ODLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>Data: DataModuleOD(...)
    ODL->>Model: ModelModuleOD(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: fit(model, datamodule)

    Trainer->>Data: setup("fit")
    Data->>Dataset: DatasetVR(...)
    Data->>Dataset: setup("fit")
    Note over Dataset: Load images from Input/<br/>Load CSV from Output/
    Dataset-->>Data: dataset_train, dataset_validate

    loop For each epoch
        Trainer->>Data: train_dataloader()
        Data->>Dataset: DataLoader(dataset_train)
        Data-->>Trainer: train_dataloader
        
        loop For each batch
            Trainer->>Data: Get batch
            Data->>Dataset: __getitem__(idx)
            Note over Dataset: Load image<br/>Load annotations<br/>Apply transforms
            Dataset-->>Data: (image, target, name)
            Data-->>Trainer: batch
            
            Trainer->>Model: training_step(batch, batch_idx)
            Model->>Model: _runModel(batch, batch_idx)
            Model->>FasterRCNN: forward(images, targets)
            FasterRCNN-->>Model: loss_dict
            Model->>Model: sum(losses)
            Model-->>Trainer: losses
            Trainer->>Trainer: backpropagation
            Trainer->>Trainer: optimizer.step()
        end
        
        Trainer->>Data: val_dataloader()
        Data-->>Trainer: val_dataloader
        
        loop For each validation batch
            Trainer->>Data: Get batch
            Data-->>Trainer: batch
            Trainer->>Model: validation_step(batch, batch_idx)
            Model->>Model: _runModel(batch, batch_idx)
            Model-->>Trainer: val_loss
        end
        
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
    participant ODL as ODLightningCLI
    participant Trainer
    participant Model as ModelModuleOD
    participant Data as DataModuleOD
    participant Dataset as DatasetVR

    User->>CLI: python main.py test --config ... --ckpt_path ...
    CLI->>ODL: ODLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>ODL: load_checkpoint()
    ODL->>Data: DataModuleOD(...)
    ODL->>Model: ModelModuleOD.load_from_checkpoint(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: test(model, datamodule)

    Trainer->>Data: setup("test")
    Data->>Dataset: DatasetVR(...)
    Data-->>Trainer: dataset_test

    Trainer->>Model: on_test_epoch_start()
    Model->>Model: _preparePrediction()
    Note over Model: Create prediction_dir<br/>Setup class colors

    Trainer->>Data: test_dataloader()
    Data-->>Trainer: test_dataloader

    loop For each batch
        Trainer->>Data: Get batch
        Data->>Dataset: __getitem__(idx)
        Dataset-->>Data: (image, target, name)
        Data-->>Trainer: batch
        
        Trainer->>Model: test_step(batch, batch_idx)
        Model->>Model: _runInference(batch, batch_idx, computeMetrics=True)
        Note over Model: model(images) - Inference mode<br/>Filter by detection_threshold<br/>Update metric_map<br/>Compute confusion matrix<br/>Save CSV predictions<br/>Save visualized images
    end

    Trainer->>Model: on_test_epoch_end()
    Model->>Model: metric_map.compute()
    Note over Model: Log mAP to TensorBoard<br/>Log confusion matrix to TensorBoard
    Model-->>Trainer: Test complete

    Trainer-->>CLI: Testing complete
    CLI-->>User: Done
```

### Prediction Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ODL as ODLightningCLI
    participant Trainer
    participant Model as ModelModuleOD
    participant Data as DataModuleOD
    participant Dataset as DatasetImage

    User->>CLI: python main.py predict --config ... --data.data_dir ...
    CLI->>ODL: ODLightningCLI()
    ODL->>ODL: parse_config()
    ODL->>ODL: load_checkpoint()
    ODL->>Data: DataModuleOD(...)
    ODL->>Model: ModelModuleOD.load_from_checkpoint(...)
    ODL->>Trainer: Trainer(...)

    ODL->>Trainer: predict(model, datamodule)

    Trainer->>Data: setup("predict")
    Data->>Dataset: DatasetImage(...)
    Data-->>Trainer: dataset_predict

    Trainer->>Model: on_predict_epoch_start()
    Model->>Model: _preparePrediction()
    Note over Model: Create prediction_dir<br/>Setup class colors

    Trainer->>Data: predict_dataloader()
    Data-->>Trainer: predict_dataloader

    loop For each batch
        Trainer->>Data: Get batch
        Data->>Dataset: __getitem__(idx)
        Note over Dataset: Load image only
        Dataset-->>Data: (image, {}, name)
        Data-->>Trainer: batch
        
        Trainer->>Model: predict_step(batch, batch_idx)
        Model->>Model: _runInference(batch, batch_idx, computeMetrics=False)
        Note over Model: model(images) - Inference mode<br/>Filter by detection_threshold<br/>Save CSV predictions<br/>Save visualized images
    end

    Model-->>Trainer: Prediction complete
    Trainer-->>CLI: Done
    CLI-->>User: Results saved in prediction/
```

---

## Folder and File Structure

### Complete Project Structure

```mermaid
graph TD
    Root["image-to-physical/"]
    
    ObjectDetection["ObjectDetection/<br/>Core Object Detection Framework"]
    ViewRecognition["ViewRecognition/<br/>View Recognition Application"]
    Utils["utils/<br/>Utility Modules"]
    
    CLI["cli.py<br/>ODLightningCLI"]
    Main["main.py<br/>CLI Entry Point"]
    MainFitTest["main_fittest.py<br/>Fit + Test"]
    DataModule["datamodule.py<br/>DataModuleOD"]
    Dataset["dataset.py<br/>DatasetOD, DatasetImage"]
    ModelModule["modelmodule.py<br/>ModelModuleOD"]
    ModelFactory["modelfactory.py<br/>Model Factory"]
    
    DatasetVR["dataset.py<br/>DatasetVR"]
    Config["config/"]
    ConfigYAML["viewrecognition.yaml"]
    ConfigLocal["viewrecognition.local.yaml"]
    
    Colors["colors.py"]
    DatasetUtil["dataset.py<br/>collate_fn"]
    Image["image.py"]
    ImageAug["imageaugmentation.py"]
    ImageViz["imagevisualization.py"]
    Lib["lib.py"]
    Metrics["metrics.py"]
    
    Notebook["ViewRecognition.ipynb"]
    Reqs["requirements.txt"]
    Readme["README.md"]
    UML["UML_DIAGRAMS.md"]
    
    Root --> ObjectDetection
    Root --> ViewRecognition
    Root --> Utils
    Root --> Notebook
    Root --> Reqs
    Root --> Readme
    Root --> UML
    
    ObjectDetection --> CLI
    ObjectDetection --> Main
    ObjectDetection --> MainFitTest
    ObjectDetection --> DataModule
    ObjectDetection --> Dataset
    ObjectDetection --> ModelModule
    ObjectDetection --> ModelFactory
    
    ViewRecognition --> DatasetVR
    ViewRecognition --> Config
    Config --> ConfigYAML
    Config --> ConfigLocal
    
    Utils --> Colors
    Utils --> DatasetUtil
    Utils --> Image
    Utils --> ImageAug
    Utils --> ImageViz
    Utils --> Lib
    Utils --> Metrics
    
    style Root fill:#e1f5ff
    style ObjectDetection fill:#fff4e1
    style ViewRecognition fill:#e8f5e9
    style Utils fill:#f3e5f5
```

### Dependency Graph

```mermaid
flowchart TD
    ExtDeps["External Dependencies<br/>• PyTorch (torch, torchvision)<br/>• PyTorch Lightning (lightning)<br/>• OpenCV (cv2)<br/>• NumPy<br/>• Pandas<br/>• Albumentations<br/>• JSONArgParse<br/>• TorchMetrics<br/>• TensorBoard"]
    
    ObjectDetection["ObjectDetection"]
    ViewRecognition["ViewRecognition"]
    Utils["utils/"]
    
    Application["ViewRecognition<br/>Application"]
    
    ExtDeps -->|used by| ObjectDetection
    ExtDeps -->|used by| ViewRecognition
    ExtDeps -->|used by| Utils
    
    ObjectDetection -->|uses| Application
    ViewRecognition -->|uses| Application
    Utils -->|uses| Application
    
    style ExtDeps fill:#e3f2fd
    style Application fill:#e8f5e9
```

---

## Module Interconnections

### Import Dependencies

```mermaid
graph LR
    subgraph ObjectDetection["ObjectDetection/"]
        CLI["cli.py"]
        Main["main.py"]
        MainFitTest["main_fittest.py"]
        DataModule["datamodule.py"]
        Dataset["dataset.py"]
        ModelModule["modelmodule.py"]
        ModelFactory["modelfactory.py"]
    end
    
    subgraph ViewRecognition["ViewRecognition/"]
        DatasetVR["dataset.py"]
    end
    
    subgraph Utils["utils/"]
        Colors["colors.py"]
        DatasetUtil["dataset.py"]
        Image["image.py"]
        ImageAug["imageaugmentation.py"]
        ImageViz["imagevisualization.py"]
        Lib["lib.py"]
        Metrics["metrics.py"]
    end
    
    subgraph External["External Libraries"]
        Lightning["lightning.pytorch.cli"]
        Torch["torch"]
        TorchVision["torchvision"]
        CV2["cv2"]
        Numpy["numpy"]
        Pandas["pandas"]
        Albumentations["albumentations"]
        TorchMetrics["torchmetrics"]
        Random["random"]
        Itertools["itertools"]
        Importlib["importlib"]
    end
    
    CLI --> Lightning
    Main --> CLI
    MainFitTest --> CLI
    DataModule --> Lightning
    DataModule --> Lib
    DataModule --> Dataset
    DataModule --> DatasetUtil
    DataModule --> ImageAug
    Dataset --> Torch
    Dataset --> CV2
    Dataset --> Numpy
    Dataset --> ImageAug
    ModelModule --> Lightning
    ModelModule --> Lib
    ModelModule --> ImageViz
    ModelModule --> Colors
    ModelModule --> TorchMetrics
    ModelModule --> Metrics
    ModelFactory --> Torch
    ModelFactory --> TorchVision
    
    DatasetVR --> Torch
    DatasetVR --> CV2
    DatasetVR --> Numpy
    DatasetVR --> Pandas
    DatasetVR --> Dataset
    DatasetVR --> ImageAug
    DatasetVR --> Image
    
    Colors --> Random
    DatasetUtil --> Itertools
    ImageAug --> CV2
    ImageAug --> Numpy
    ImageAug --> Albumentations
    ImageViz --> CV2
    ImageViz --> Numpy
    ImageViz --> Colors
    Lib --> Importlib
    Metrics --> Numpy
    
    style ObjectDetection fill:#fff4e1
    style ViewRecognition fill:#e8f5e9
    style Utils fill:#f3e5f5
    style External fill:#e3f2fd
```

### Data Flow Between Modules

```mermaid
flowchart TD
    ConfigLayer["Configuration Layer<br/>(viewrecognition.yaml +<br/>viewrecognition.local.yaml)"]
    
    CLILayer["CLI Layer<br/>(ODLightningCLI, main.py,<br/>main_fittest.py)"]
    
    DataModuleOD["DataModuleOD"]
    ModelModuleOD["ModelModuleOD"]
    Trainer["Trainer"]
    
    DatasetVR["DatasetVR"]
    FasterRCNN["FasterRCNN"]
    
    UtilsModules["Utility Modules<br/>(imageaugmentation, image,<br/>metrics, visualization,<br/>colors, dataset)"]
    
    Execution["Training/Test/<br/>Prediction<br/>Execution"]
    
    ConfigLayer -->|config| CLILayer
    CLILayer -->|creates| DataModuleOD
    CLILayer -->|creates| ModelModuleOD
    CLILayer -->|creates| Trainer
    
    DataModuleOD -->|uses| DatasetVR
    ModelModuleOD -->|uses| FasterRCNN
    Trainer -->|manages| DataModuleOD
    Trainer -->|manages| ModelModuleOD
    
    DatasetVR -->|uses| UtilsModules
    FasterRCNN -->|uses| UtilsModules
    
    DataModuleOD --> Execution
    ModelModuleOD --> Execution
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
    ImageFiles["Image Files<br/>(Input/)"]
    CSVFiles["CSV Files<br/>(Output/)"]
    
    DatasetVR["DatasetVR.__getitem__()<br/>• Load image (cv2.imread)<br/>• Load CSV annotations (pandas)<br/>• Filter by MainLabel='View'<br/>• Map SubLabels to classes<br/>• Convert center/wh to bbox coordinates<br/>• Scale to resized dimensions<br/>• Convert to torch tensors"]
    
    PreProcess["preProcess() (utils)<br/>• BGR → RGB → Grayscale<br/>• Binary threshold<br/>• Resize<br/>• Normalize"]
    
    GetTransform["getTransform() (Training)<br/>• Augmentations (crop, rotate, etc.)<br/>• ToTensor"]
    
    CollateFn["collate_fn() (utils)<br/>• Handle variable-length batches"]
    
    TransferDevice["DataModuleOD.transfer_batch_to_device()<br/>• Move to GPU/CPU"]
    
    TrainingStep["ModelModuleOD.training_step()<br/>• Forward pass (FasterRCNN)<br/>• Compute loss<br/>• Backward pass"]
    
    ImageFiles --> DatasetVR
    CSVFiles --> DatasetVR
    DatasetVR --> PreProcess
    PreProcess --> GetTransform
    GetTransform --> CollateFn
    CollateFn --> TransferDevice
    TransferDevice --> TrainingStep
    
    style ImageFiles fill:#e1f5ff
    style CSVFiles fill:#e1f5ff
    style TrainingStep fill:#e8f5e9
```

### Inference Data Flow

```mermaid
flowchart TD
    ImageFiles["Image Files<br/>(Input/)"]
    
    DatasetGetItem["DatasetVR.__getitem__() or<br/>DatasetImage.__getitem__()<br/>• Load image<br/>• (Optional: Load annotations)"]
    
    PreProcess["preProcess()<br/>• BGR → RGB → Grayscale<br/>• Binary threshold<br/>• Resize<br/>• Normalize"]
    
    GetNoTransform["getNoTransform()<br/>• ToTensor only"]
    
    RunInference["ModelModuleOD._runInference()<br/>• Model forward (inference mode)<br/>• Filter by detection_threshold"]
    
    PostProcessing["Post-processing<br/>• (Test only) Compute metrics<br/>• Save CSV predictions<br/>• Visualize with bounding boxes<br/>• Save images"]
    
    ImageFiles --> DatasetGetItem
    DatasetGetItem --> PreProcess
    PreProcess --> GetNoTransform
    GetNoTransform --> RunInference
    RunInference --> PostProcessing
    
    style ImageFiles fill:#e1f5ff
    style PostProcessing fill:#fff4e1
```

---

## Configuration Flow

```mermaid
flowchart TD
    BaseConfig["viewrecognition.yaml<br/>(Base Configuration)"]
    
    LocalConfig["viewrecognition.local.yaml<br/>(Local Overrides)"]
    
    Parser["JSONArgParse / LightningCLI"]
    
    TrainerConfig["Trainer<br/>Config"]
    ModelConfig["Model<br/>Config"]
    DataConfig["Data<br/>Config"]
    
    Execution["PyTorch Lightning<br/>Execution"]
    
    BaseConfig -->|merged with| LocalConfig
    LocalConfig -->|parsed by| Parser
    Parser -->|creates| TrainerConfig
    Parser -->|creates| ModelConfig
    Parser -->|creates| DataConfig
    TrainerConfig -->|instantiates| Execution
    ModelConfig -->|instantiates| Execution
    DataConfig -->|instantiates| Execution
    
    style BaseConfig fill:#e3f2fd
    style LocalConfig fill:#fff3e0
    style Execution fill:#e8f5e9
```

---

## Summary

This document provides comprehensive diagrams and design documentation for the Image-to-Physical Object Detection project. The system follows a modular, extensible architecture based on PyTorch Lightning, making it easy to:

1. **Add new object detection tasks** by creating new dataset classes
2. **Switch models** via configuration without code changes
3. **Extend functionality** through the utility modules
4. **Configure workflows** through YAML files

The design separates concerns:
- **ObjectDetection/**: Reusable framework components
- **ViewRecognition/**: Application-specific implementation
- **utils/**: Shared utilities

This separation allows the framework to be reused for other object detection tasks while keeping application-specific code isolated.
