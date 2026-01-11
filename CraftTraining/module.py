import os
import cv2
import json
import math
import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
import pandas as pd
from typing import Any, Optional
from utils.craftImagevisualization import drawCV2BBWithText, visualizeOneBatchImages
from utils.craftmetrics import CraftMetrics
from utils.craft_postprocess import craft_watershed

PREDICTION_OUTPUT_FOLDER = "prediction"

class CraftLightningModule(L.LightningModule):
    def __init__(
        self,
        save_dir: str,
        name: str,
        visualize_training_images: bool,
        save_predicted_images: bool = True,
        model: Optional[Any] = None,
        pretrained_model: str = "",
        ios_threshold: float = 0.05,
        debug_metrics: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.metrics_manager = CraftMetrics(
            ios_threshold=ios_threshold,
            debug=debug_metrics
        )
        self.visualize_training_images = visualize_training_images
        self.save_predicted_images = save_predicted_images
        self.save_dir = save_dir
        self.name = name
        self.prediction_dir = None
        self.test_cfg = {}
        self.pretrained_model = pretrained_model
        self.model = model


        # if pretrained_model:
        #     ckpt = torch.load(pretrained_model, map_location="cpu")
        #     if "state_dict" in ckpt:
        #         self.model.load_state_dict(ckpt["state_dict"], strict=False)
        #     else:
        #         self.model.load_state_dict(ckpt, strict=False)
        #     print(f"Loaded pretrained model from: {pretrained_model}")

    def _compute_loss(self, outputs, targets):
        """Compute BCE loss for region + affinity maps."""
        region_logit = outputs["region_logit"]
        affinity_logit = outputs["affinity_logit"]
        total_loss = 0.0

        for i in range(len(targets)):
            t = targets[i]
            region_t = t["region"].unsqueeze(0).to(self.device)
            affinity_t = t["affinity"].unsqueeze(0).to(self.device)

            r_logit = F.interpolate(region_logit[i:i+1], size=region_t.shape[-2:], mode="bilinear", align_corners=False)
            a_logit = F.interpolate(affinity_logit[i:i+1], size=affinity_t.shape[-2:], mode="bilinear", align_corners=False)

            loss_r = F.binary_cross_entropy_with_logits(r_logit, region_t)
            loss_a = F.binary_cross_entropy_with_logits(a_logit, affinity_t)
            total_loss += (loss_r + loss_a)

        # print(f"Batch loss: {total_loss.item()/len(targets):.4f}")

        return total_loss / len(targets)

    def _runModel(self, batch, batch_idx):
        """Forward + loss wrapper."""
        images, targets, _ = batch
        # print(f"\n[DEBUG] image is on device: {images[0].device}")

        outputs = self.model(images)
        loss = self._compute_loss(outputs, targets)
        return loss

   


    def on_fit_start(self):
        if self.pretrained_model:
            ckpt = torch.load(self.pretrained_model, map_location=self.device)
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
            print("✓ Pretrained weights loaded at fit start (after resume check)")





    #  Visualization of first batch (optional sanity check)
    def on_train_start(self):

        print(f"\n\n🚀 Training started on device: {self.device}")
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(self.device.index)}")
        else:
            print("⚠️ CUDA NOT available — running on CPU!")
        print("\n") 

        return super().on_train_start()

    def on_train_batch_start(self, batch, batch_idx):
        """Visualize first batch at epoch 0."""
        if self.visualize_training_images and self.trainer.current_epoch == 0 and batch_idx == 0:
            visualizeOneBatchImages(batch)
        return super().on_train_batch_start(batch, batch_idx)


    #  Training + Validation

    def training_step(self, batch, batch_idx):
        images, targets, names = batch
        # print(f"\n[DEBUG] image is on device: {images[0].device}")


        # if batch_idx == 0:
            # print(f"\n\n[DEBUG] batch type: {type(batch)}")
            # print(f"[DEBUG] image[0] type: {type(images[0])}, device: {images[0].device}")
            # print(f"[DEBUG] type(images): {type(images)}")
            # if isinstance(images, torch.Tensor):
            #     print(f"[DEBUG] images shape: {images.shape}\n")
            # elif isinstance(images, list):
            #     print(f"[DEBUG] list of {len(images)} images, first shape: {images[0].shape}\n")

        loss = self._runModel(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._runModel(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def _preparePrediction(self):
        base_dir = self.logger.log_dir if self.logger else os.path.join(self.save_dir, self.name)
        self.prediction_dir = os.path.join(base_dir, PREDICTION_OUTPUT_FOLDER)
        os.makedirs(self.prediction_dir, exist_ok=True)

    #  Load test configuration thresholds from YAML
    def _load_test_cfg(self):
        """Load test thresholds from DataModule (YAML-driven)."""
        try:
            dm = getattr(self.trainer, "datamodule", None)
            cfg = getattr(dm, "test_cfg", {}) or {}
            self.test_cfg = dict(cfg)
        except Exception:
            self.test_cfg = {}

        # defaults
        self.test_cfg.setdefault("text_threshold", 0.7)
        self.test_cfg.setdefault("link_threshold", 0.4)
        self.test_cfg.setdefault("low_text", 0.4)
        self.test_cfg.setdefault("vis_opt", False)

        print(f"\n[INFO] Loaded test thresholds: {self.test_cfg}")


 
    def _runInference(self, batch, batch_idx):
        images, targets, names = batch

        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        outputs = self.model(images)
        region_score_t = torch.sigmoid(outputs["region_logit"])
        affinity_score_t = torch.sigmoid(outputs["affinity_logit"])

        text_threshold = self.test_cfg.get("text_threshold", 0.7)
        link_threshold = self.test_cfg.get("link_threshold", 0.4)
        low_text = self.test_cfg.get("low_text", 0.4)
        min_area = int(self.test_cfg.get("min_area", 10))

        for i, name in enumerate(names):
            reg_np = region_score_t[i, 0].detach().cpu().numpy()
            aff_np = affinity_score_t[i, 0].detach().cpu().numpy()

            polys_and_scores = craft_watershed(
                reg_np,
                aff_np,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                min_area=min_area,
                debug_save=None
            )

            t = targets[i]
            scale = float(t.get("scale", 1.0))
            x_off, y_off = t.get("pad_offset", (0, 0))
            orig_w, orig_h = t["orig_size"]

            pred_polys = []
            pred_scores = []
            for quad, score in polys_and_scores:
                q = np.array(quad, dtype=float).copy()
                q[:, 0] -= x_off
                q[:, 1] -= y_off
                if scale != 0 and not math.isclose(scale, 1.0):
                    q *= 1.0 / scale
                q[:, 0] = np.clip(q[:, 0], 0, orig_w - 1)
                q[:, 1] = np.clip(q[:, 1], 0, orig_h - 1)

                pred_polys.append(q.tolist())
                pred_scores.append(float(score))

            out_txt_path = os.path.join(self.prediction_dir, os.path.splitext(name)[0] + ".txt")
            with open(out_txt_path, "w", encoding="utf-8") as fh:
                for p, s in zip(pred_polys, pred_scores):
                    coords = ",".join([f"{float(x):.2f}" for xy in p for x in xy])
                    fh.write(f"{coords},{s:.6f}\n")

            gt_polys = t.get("polys") or []
            self.metrics_manager.update(preds=pred_polys, gts=gt_polys, img_id=name)

            if self.save_predicted_images:
                img_path = self._resolve_image_path(name)
                if img_path and os.path.exists(img_path):
                    img = cv2.imread(img_path)
                else:
                    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

                for poly, score in zip(pred_polys, pred_scores):
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    x0, y0 = int(poly[0][0]), int(poly[0][1])
                    cv2.putText(img, f"{score:.2f}", (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imwrite(os.path.join(self.prediction_dir, name), img)


    # Helper to locate original test image
    def _resolve_image_path(self, name: str):
        dm = getattr(self.trainer, "datamodule", None)
        if dm:
            for sub in ["Testing/Input", "Validation/Input", "Training/Input"]:
                path = os.path.join(dm.data_dir, sub, name)
                if os.path.exists(path):
                    return path
        return None


    #  Hooks for testing and prediction
    def on_test_epoch_start(self):
        self._preparePrediction()
        self._load_test_cfg()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        self._runInference(batch, batch_idx)
        return

    def on_test_epoch_end(self):
        summary = self.metrics_manager.compute()

        tb = self.logger.experiment
        tb.add_text("Metrics/Summary", json.dumps(summary, indent=2))

        print("\n=== Detection Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        # Reset for next evaluation run
        self.metrics_manager.reset()

        return super().on_test_epoch_end()



    def on_predict_epoch_start(self):
        self._preparePrediction()
        self._load_test_cfg()
        return super().on_predict_epoch_start()

    def predict_step(self, batch, batch_idx):
        self._runInference(batch, batch_idx)
        return
