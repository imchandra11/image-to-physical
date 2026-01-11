import os
from typing import Optional
import lightning as L
import utils.lib
from utils.imagevisualization import drawCV2BBWithText, visualizeOneBatchImages
from utils.colors import getRandomBASEColors, getRandomTABLEAUColors, hex_to_bgr
import cv2
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.metrics import computeBBConfusionMatrix
from pprint import pprint
import json
import pandas as pd
from tabulate import tabulate

PREDICTION_OUTPUT_FOLDER="prediction"

class ModelModuleOD(L.LightningModule):
    def __init__(self, 
        num_classes: int,
        detection_threshold: float,
        visualize_training_images: bool,
        save_predicted_images: bool,
        torch_model: Optional[dict] = None,
        torch_model_factory: Optional[dict] = None
    ) -> None:
        r"""Model for object detection

        Args:
            num_classes: Number of classes
            detection_threshold: The detection threshold to use
            visualize_training_images: Show first batch transformed images before starting training
            save_predicted_images: Save the image predictions
            torch_model: Custom CNN model to use
            torch_model_factory: Function to run to get CNN model to use
        """
        super().__init__()

        callable_model, init_args_model = utils.lib.getCallableAndArgs(torch_model, torch_model_factory)
        self.model = callable_model(num_classes=num_classes, **(init_args_model or {}))

        self.num_classes = num_classes
        self.confusion_matrix = [[0,0,0,0] for _ in range(num_classes)]
        self.detection_threshold = detection_threshold
        self.visualize_training_images = visualize_training_images
        self.save_predicted_images = save_predicted_images
        self.metric_map = MeanAveragePrecision(class_metrics=True)
        self.save_hyperparameters()

    def _runModel(self, batch, batch_idx):
        images, targets, names = batch
        loss_dict = self.model(images, targets)  
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        return (losses, loss_value)

    def on_train_batch_start(self, batch, batch_idx):
        if (self.visualize_training_images):
            if self.trainer.current_epoch == 0 and batch_idx == 0:
                visualizeOneBatchImages(batch)
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        losses, loss_value = self._runModel(batch, batch_idx)
        self.log("train_loss", loss_value, prog_bar=True, logger=True)
        return losses

    def validation_step(self, batch, batch_idx):
        # https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn
        # Put the model in train mode for loss computation (specific for PyTorch CNN implementation)
        self.model.train()
        losses, loss_value = self._runModel(batch, batch_idx)
        # Put the model back in eval mode (specific for PyTorch CNN implementation)
        self.model.eval()
        self.log("val_loss", loss_value, prog_bar=True, logger=True)

    def _preparePrediction(self):
        self.prediction_dir = os.path.join(self.trainer.log_dir, PREDICTION_OUTPUT_FOLDER)
        os.makedirs(self.prediction_dir, exist_ok = True)
        if (self.save_predicted_images):    
            classes = dict(self.trainer.datamodule.classes)
            # colorpalette = getRandomTABLEAUColors(len(classes))
            # self.classcolors = {str(v): hex_to_bgr(colorpalette[index]) for index, (k, v) in enumerate(classes.items())}
            colorpalette = getRandomBASEColors(len(classes))
            self.classcolors = {str(v): colorpalette[index] for index, (k, v) in enumerate(classes.items())}
            # print(f'Prediction using colors {self.classcolors}')

    def _runInference(self, batch, batch_idx, computeMetrics = False):
        images, targets, names = batch

        outputs = self.model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        if computeMetrics:
            self.metric_map.update(outputs,targets)

        for  ii in range(len(images)):
            image = images[ii]
            name = names[ii]
            target = targets[ii]
            output = outputs[ii]

            output = {key: value.data.numpy() for key, value in output.items() if key in ['boxes', 'scores', 'labels']}
            target = {key: value.data.numpy() for key, value in target.items() if key in ['boxes', 'scores', 'labels']}

            detection_results = {
                'boxes' : output['boxes'][output['scores'] >= self.detection_threshold].astype(np.int32),
                'labels' : output['labels'][output['scores'] >= self.detection_threshold],
                'scores' : output['scores'][output['scores'] >= self.detection_threshold]
            }




            if target:
                boxes = np.concatenate([target['boxes'], detection_results["boxes"]], axis=0)
                scores = np.concatenate([-np.ones((len(target["boxes"]))), detection_results['scores']], axis=0)
                labels = np.concatenate([target['labels'], detection_results["labels"]], axis=0)
            else:
                boxes =  detection_results["boxes"]
                scores = detection_results['scores']
                labels = detection_results["labels"]
            df = pd.DataFrame({
                "Left":boxes[:,0],
                "Top":boxes[:,1],
                "Right":boxes[:,2],
                "Bot":boxes[:,3],
                "Score":scores,
                "Label":labels
            })
            df.to_csv(os.path.join(self.prediction_dir, os.path.splitext(name)[0]+".csv"))

            if computeMetrics:
                metric = computeBBConfusionMatrix(target, detection_results, self.num_classes)
                self.confusion_matrix = [[a + b for a, b in zip(cl, ml)] for cl, ml in zip(self.confusion_matrix, metric)]

            if (self.save_predicted_images):
                draw_boxes = detection_results['boxes'].copy()

                image = image.cpu().numpy()[0]
                image = image*255

                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for box_num, box in enumerate(draw_boxes):
                    class_name = str(detection_results['labels'][box_num])
                    score = str(round(detection_results['scores'][box_num],3))
                    drawCV2BBWithText(image, box, class_name + " " + score, self.classcolors[class_name])
                cv2.imwrite(os.path.join(self.prediction_dir, name), image)

    def on_test_epoch_start(self):
        self._preparePrediction()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        self._runInference(batch, batch_idx, computeMetrics = True)
        return

    def on_test_epoch_end(self):
        mmc = self.metric_map.compute()
        map_jsonable = {}
        for key, value in mmc.items() :
            if value.dim() == 0 :
                map_jsonable[key] = value.item()
            else :
                map_jsonable[key] = [x.item() for x in value]
        json_string = json.dumps(map_jsonable, indent=1)
        tensorboard = self.logger.experiment
        tensorboard.add_text("MeanAveragePrecision", json_string)

        cmt = tabulate(tabular_data=self.confusion_matrix, 
            showindex=True,
            headers=['Label', 'TP', 'FP', 'FN', 'TN'], 
            tablefmt="simple")
        tensorboard.add_text("ConfusionMatrix", cmt)

        # self.metric_map.plot()

        return super().on_test_epoch_end()

    def on_predict_epoch_start(self):
        self._preparePrediction()
        return super().on_predict_epoch_start()

    def predict_step(self, batch, batch_idx):
        self._runInference(batch, batch_idx, computeMetrics = False)
        return

    # Done in CLI yaml    
    # def configure_optimizers(self):
    #     params = [p for p in self.model.parameters() if p.requires_grad]

    #     optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
    #                               amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, 
    #                               fused=None)
    #     # total_steps=epochs
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=5, pct_start=0.1)

    #     # Default scheduler.step is per epoch
    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": scheduler,
    #     }
    # }


