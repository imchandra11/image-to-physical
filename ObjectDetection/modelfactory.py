import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def getModelfasterrcnn_resnet50_fpn(
    num_classes: int,
    pretrained: bool,
    **kwargs
) -> FasterRCNN:
    r"""Get the model fasterrcnn_resnet50_fpn
        Args:
            num_classes: Number of classes
            pretrained: Use DEFAULT weights
    """
    if(pretrained):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.backbone.body.layer1.append(torch.nn.Dropout(p=0.2))
    return model
