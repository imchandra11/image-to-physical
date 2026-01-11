import torch.nn as nn
from EasyOCRTraining.utilities.transformation import TPS_SpatialTransformerNetwork
from EasyOCRTraining.utilities.feature_extraction import ResNet_FeatureExtractor, VGG_FeatureExtractor, RCNN_FeatureExtractor
from EasyOCRTraining.utilities.sequence_modeling import BidirectionalLSTM
from EasyOCRTraining.utilities.prediction import Attention

class EasyOCRNet(nn.Module):
    def __init__(
        self,
        Transformation: str = "None",
        FeatureExtraction: str = "VGG",
        SequenceModeling: str = "BiLSTM",
        Prediction: str = "CTC",
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 256,
        num_fiducial: int = 20,
        imgH: int = 32,
        imgW: int = 100,
        num_class: int = 38  # default placeholder
    ):
        super().__init__()
        self.stages = {
            "Trans": Transformation,
            "Feat": FeatureExtraction,
            "Seq": SequenceModeling,
            "Pred": Prediction
        }

        # Transformation
        if Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel
            )
        else:
            self.Transformation = None

        # FeatureExtraction
        if FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception("No FeatureExtraction module selected")

        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # Sequence modeling
        if SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(output_channel, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
            )
            self.SequenceModeling_output = hidden_size
        else:
            self.SequenceModeling = None
            self.SequenceModeling_output = output_channel

        # Prediction
        if Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        elif Prediction == "Attn":
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)

    def forward(self, input, text=None, is_train=True):
        # Transformation stage
        if self.Transformation:
            input = self.Transformation(input)

        # Feature extraction
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling
        if self.SequenceModeling:
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        # Prediction
        if self.stages["Pred"] == "CTC":
            return self.Prediction(contextual_feature)
        else:
            return self.Prediction(contextual_feature, text, is_train, batch_max_length=25)
