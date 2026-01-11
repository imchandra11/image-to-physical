# import os
# import csv
# import lightning as L
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
# from EasyOCRTraining.utilities.labelconverter import CTCLabelConverter, AttnLabelConverter
# from EasyOCRTraining.model import EasyOCRNet


# class EasyOCRLitModule(L.LightningModule):
#     def __init__(
#         self,
#         character: str,
#         pretrained_model:str,
#         Prediction: str = "CTC",
#         batch_max_length: int = 25,
#         model: EasyOCRNet = None,
#         save_predicted_images: bool = True,
#         decode: str = "greedy",
#         freeze_FeatureExtraction: bool = False,
#         freeze_SequenceModeling: bool = False,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.pretrained_model = pretrained_model
#         self.Prediction = Prediction
#         self.character = character
#         self.batch_max_length = batch_max_length
#         self.save_predicted_images = save_predicted_images
#         self.decode = decode
#         self.freeze_FeatureExtraction = freeze_FeatureExtraction
#         self.freeze_SequenceModeling = freeze_SequenceModeling

#         if Prediction == "CTC":
#             self.num_class = len(character) + 1
#             self.converter = CTCLabelConverter(character)
#             self.criterion = nn.CTCLoss(zero_infinity=True)
#         else:
#             self.num_class = len(character) + 2
#             self.converter = AttnLabelConverter(character)
#             self.criterion = nn.CrossEntropyLoss(ignore_index=0)

#         self.model = model
#         self.prediction_dir = None
#         self.prediction_results = []

#         # Freeze logic
#         if freeze_FeatureExtraction:
#             if hasattr(self.model, 'FeatureExtraction'):
#                 for param in self.model.FeatureExtraction.parameters():
#                     param.requires_grad = False

#         if freeze_SequenceModeling:
#             if hasattr(self.model, 'SequenceModeling'):
#                 for param in self.model.SequenceModeling.parameters():
#                     param.requires_grad = False

#         if pretrained_model:
#             ckpt = torch.load(pretrained_model, map_location="cpu")
#             if "state_dict" in ckpt:
#                 self.model.load_state_dict(ckpt["state_dict"], strict=False)
#             else:
#                 self.model.load_state_dict(ckpt, strict=False)
#             print(f"Loaded pretrained model from: {pretrained_model}")

#     def forward(self, images, text=None):
#         return self.model(images, text, is_train=self.training)

#     def training_step(self, batch, batch_idx):
#         images, labels = batch
#         text_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
#         batch_size = images.size(0)

#         if self.Prediction == "CTC":
#             preds = self(images, text_input).log_softmax(2).permute(1, 0, 2)
#             preds_size = torch.IntTensor([preds.size(0)] * batch_size)
#             loss = self.criterion(preds, text_input.to(self.device), preds_size.to(self.device), length.to(self.device))
#         else:
#             preds = self(images, text_input[:, :-1])
#             target = text_input[:, 1:].contiguous().view(-1)
#             loss = self.criterion(preds.view(-1, preds.shape[-1]), target.to(self.device))

#         self.log("train_loss", loss, on_step=True, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, labels = batch
#         text_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
#         batch_size = images.size(0)

#         if self.Prediction == "CTC":
#             preds = self(images, text_input).log_softmax(2).permute(1, 0, 2)
#             preds_size = torch.IntTensor([preds.size(0)] * batch_size)
#             loss = self.criterion(preds, text_input.to(self.device), preds_size.to(self.device), length.to(self.device))
#             _, preds_index = preds.permute(1, 0, 2).max(2)
#             # preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
#             if self.decode == "greedy":
#                 preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
#             else:
#                 preds_str = self.converter.decode_beamsearch(preds.permute(1, 0, 2).cpu().detach().numpy())
#         else:
#             preds = self(images, text_input[:, :-1], is_train=False)
#             preds = preds[:, :text_input.shape[1]-1, :]
#             target = text_input[:, 1:].contiguous().view(-1)
#             loss = self.criterion(preds.view(-1, preds.shape[-1]), target.to(self.device))
#             _, preds_index = F.softmax(preds, dim=2).max(2)
#             preds_str = self.converter.decode(preds_index, torch.IntTensor([l + 1 for l in length]))

#         correct = sum(pred == gt for pred, gt in zip(preds_str, labels))
#         acc = correct / batch_size * 100.0
#         self.log("val_loss", loss, on_epoch=True)
#         self.log("val_acc", acc, on_epoch=True)
#         return {"val_loss": loss, "val_acc": acc}

#     def _preparePrediction(self):
#         self.prediction_dir = os.path.join(self.trainer.log_dir, "prediction")
#         os.makedirs(self.prediction_dir, exist_ok=True)
#         self.prediction_results = []

#     def _runInference(self, batch):
#         images, labels = batch
#         batch_size = images.size(0)

#         if self.Prediction == "CTC":
#             preds = self(images, None).log_softmax(2).permute(1, 0, 2)
#             preds_size = torch.IntTensor([preds.size(0)] * batch_size)
#             _, preds_index = preds.permute(1, 0, 2).max(2)
#             # preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
#             if self.decode == "greedy":
#                 preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
#             else:
#                 preds_str = self.converter.decode_beamsearch(preds.permute(1, 0, 2).cpu().detach().numpy())

#         else:
#             go_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
#             preds = self(images, go_input[:, :-1], is_train=False)
#             preds = preds[:, :go_input.shape[1]-1, :]
#             _, preds_index = F.softmax(preds, dim=2).max(2)
#             preds_str = self.converter.decode(preds_index, torch.IntTensor([l + 1 for l in length]))

#         for label, pred in zip(labels, preds_str):
#             self.prediction_results.append({"ground_truth": label, "predicted_text": pred})

#     def _writePredictionCSV(self, filename: str):
#         filepath = os.path.join(self.prediction_dir, filename)
#         with open(filepath, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=["ground_truth", "predicted_text"])
#             writer.writeheader()
#             writer.writerows(self.prediction_results)

#     def on_test_epoch_start(self):
#         self._preparePrediction()

#     def test_step(self, batch, batch_idx):
#         self._runInference(batch)

#     def on_test_epoch_end(self):
#         # self._writePredictionCSV(f"test_epoch{self.current_epoch}.csv")
#         self._writePredictionCSV(f"output.csv")

#     def on_predict_epoch_start(self):
#         self._preparePrediction()

#     def predict_step(self, batch, batch_idx):
#         self._runInference(batch)

#     def on_predict_epoch_end(self):
#         self._writePredictionCSV(f"predict_epoch{self.current_epoch}.csv")


import os
import csv
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from EasyOCRTraining.utilities.labelconverter import CTCLabelConverter, AttnLabelConverter
from EasyOCRTraining.model import EasyOCRNet
from utils.easyocrmetrics import EasyOCRMetricsManager


class EasyOCRLitModule(L.LightningModule):
    def __init__(
        self,
        character: str,
        pretrained_model: str,
        Prediction: str = "CTC",
        batch_max_length: int = 25,
        model: EasyOCRNet = None,
        save_predicted_images: bool = True,
        decode: str = "greedy",
        freeze_FeatureExtraction: bool = False,
        freeze_SequenceModeling: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model = pretrained_model
        self.Prediction = Prediction
        self.character = character
        self.batch_max_length = batch_max_length
        self.save_predicted_images = save_predicted_images
        self.decode = decode
        self.freeze_FeatureExtraction = freeze_FeatureExtraction
        self.freeze_SequenceModeling = freeze_SequenceModeling

        if Prediction == "CTC":
            self.num_class = len(character) + 1
            self.converter = CTCLabelConverter(character)
            self.criterion = nn.CTCLoss(zero_infinity=True)
        else:
            self.num_class = len(character) + 2
            self.converter = AttnLabelConverter(character)
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model = model
        self.prediction_dir = None
        self.prediction_results = []
        self.metrics_manager = EasyOCRMetricsManager()

        if freeze_FeatureExtraction and hasattr(self.model, 'FeatureExtraction'):
            for param in self.model.FeatureExtraction.parameters():
                param.requires_grad = False

        if freeze_SequenceModeling and hasattr(self.model, 'SequenceModeling'):
            for param in self.model.SequenceModeling.parameters():
                param.requires_grad = False

        if pretrained_model:
            ckpt = torch.load(pretrained_model, map_location="cpu")
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained model from: {pretrained_model}")

    def forward(self, images, text=None):
        return self.model(images, text, is_train=self.training)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        text_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
        batch_size = images.size(0)

        if self.Prediction == "CTC":
            preds = self(images, text_input).log_softmax(2).permute(1, 0, 2)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = self.criterion(preds, text_input.to(self.device), preds_size.to(self.device), length.to(self.device))
        else:
            preds = self(images, text_input[:, :-1])
            target = text_input[:, 1:].contiguous().view(-1)
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.to(self.device))

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        text_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
        batch_size = images.size(0)

        if self.Prediction == "CTC":
            preds = self(images, text_input).log_softmax(2).permute(1, 0, 2)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = self.criterion(preds, text_input.to(self.device), preds_size.to(self.device), length.to(self.device))
            _, preds_index = preds.permute(1, 0, 2).max(2)
            if self.decode == "greedy":
                preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
            else:
                preds_str = self.converter.decode_beamsearch(preds.permute(1, 0, 2).cpu().detach().numpy())
        else:
            preds = self(images, text_input[:, :-1], is_train=False)
            preds = preds[:, :text_input.shape[1]-1, :]
            target = text_input[:, 1:].contiguous().view(-1)
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.to(self.device))
            _, preds_index = F.softmax(preds, dim=2).max(2)
            preds_str = self.converter.decode(preds_index, torch.IntTensor([l + 1 for l in length]))

        self.metrics_manager.update(labels, preds_str)
        correct = sum(pred == gt for pred, gt in zip(preds_str, labels))
        acc = correct / batch_size * 100.0
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        summary = self.metrics_manager.summarize()
        self.log("val_wer", summary.get("wer", 0.0))
        self.log("val_cer", summary.get("cer", 0.0))
        if self.logger:
            tb = self.logger.experiment
            tb.add_text("Metrics/WER_CER", f"WER: {summary['wer']:.4f}, CER: {summary['cer']:.4f}", self.current_epoch)
        self.metrics_manager.reset()

    def _preparePrediction(self):
        self.prediction_dir = os.path.join(self.trainer.log_dir, "prediction")
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.prediction_results = []

    def _runInference(self, batch):
        images, labels = batch
        batch_size = images.size(0)

        if self.Prediction == "CTC":
            preds = self(images, None).log_softmax(2).permute(1, 0, 2)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            if self.decode == "greedy":
                preds_str = self.converter.decode_greedy(preds_index.reshape(-1).cpu(), preds_size)
            else:
                preds_str = self.converter.decode_beamsearch(preds.permute(1, 0, 2).cpu().detach().numpy())
        else:
            go_input, length = self.converter.encode(labels, batch_max_length=self.batch_max_length)
            preds = self(images, go_input[:, :-1], is_train=False)
            preds = preds[:, :go_input.shape[1]-1, :]
            _, preds_index = F.softmax(preds, dim=2).max(2)
            preds_str = self.converter.decode(preds_index, torch.IntTensor([l + 1 for l in length]))

        for label, pred in zip(labels, preds_str):
            self.prediction_results.append({"ground_truth": label, "predicted_text": pred})

    def _writePredictionCSV(self, filename: str):
        filepath = os.path.join(self.prediction_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ground_truth", "predicted_text"])
            writer.writeheader()
            writer.writerows(self.prediction_results)

    def on_test_epoch_start(self):
        self._preparePrediction()

    def test_step(self, batch, batch_idx):
        self._runInference(batch)

    def on_test_epoch_end(self):
        self._writePredictionCSV(f"output.csv")

    def on_predict_epoch_start(self):
        self._preparePrediction()

    def predict_step(self, batch, batch_idx):
        self._runInference(batch)

    def on_predict_epoch_end(self):
        self._writePredictionCSV(f"predict_epoch{self.current_epoch}.csv")
