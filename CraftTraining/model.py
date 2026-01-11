"""
CRAFT network with VGG16-BN backbone and configurable freezing.
Usage:
    model = CraftNet(in_ch, backbone='vgg16_bn', pretrained=True, freeze_until='conv2_2')
    outs = model(images)  # outs is dict with 'region_logit' and 'affinity_logit'
"""
from typing import Optional, Dict
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _conv_relu_conv(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )

class CraftNet(nn.Module):
    def __init__(
        self,
        in_ch : int = 3,
        backbone: str = "vgg16_bn",
        pretrained: bool = True,
        freeze_until: Optional[str] = "conv2_2",
        feature_extract: bool = False,
        head_channels: int = 128,
    ):
        """
        Args:
            backbone: "vgg16_bn" (default). (Extensible later.)
            pretrained: load ImageNet weights for backbone if True.
            freeze_until: string like "conv2_2" or "none". Freezes backbone stages up to that block.
                          Implementation freeze rule: parse `convN` and freeze stages [1..N].
                          Example: "conv2_2" -> freeze stage1 and stage2.
            feature_extract: if True, freeze entire backbone (only head will train).
            head_channels: internal channels in fusion head (default 128).
        """
        super().__init__()
        self.in_ch = in_ch
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze_until = freeze_until
        self.feature_extract = feature_extract
        self.head_channels = head_channels

        # BUILD BACKBONE (currently supports vgg16_bn)
        if backbone != "vgg16_bn":
            raise ValueError("Currently only 'vgg16_bn' backbone is supported. Add others as needed.")

        vgg = models.vgg16_bn(pretrained=pretrained).features

        # VGG16-BN features are a long sequential module. We split into 5 stages.
        # These slice indices are common for vgg16_bn (conv blocks + pool layers).
        # stage1: conv1_1 .. conv1_2 (+pool)
        # stage2: conv2_1 .. conv2_2 (+pool)
        # stage3: conv3_1 .. conv3_3 (+pool)
        # stage4: conv4_1 .. conv4_3 (+pool)
        # stage5: conv5_1 .. conv5_3 (+pool)
        self.stage1 = nn.Sequential(*list(vgg.children())[:6])    # up to first MaxPool
        self.stage2 = nn.Sequential(*list(vgg.children())[6:13])
        self.stage3 = nn.Sequential(*list(vgg.children())[13:23])
        self.stage4 = nn.Sequential(*list(vgg.children())[23:33])
        self.stage5 = nn.Sequential(*list(vgg.children())[33:43])

        # Lateral convs to reduce channels to a fixed size for fusion
        # VGG channels: stage1->64, stage2->128, stage3->256, stage4->512, stage5->512
        hc = head_channels
        self.conv5 = nn.Sequential(nn.Conv2d(512, hc, 3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(512, hc, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(256, hc, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, hc // 2, 3, padding=1), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, hc // 2, 3, padding=1), nn.ReLU(inplace=True))

        # Merge convs after concatenation
        self.merge54 = _conv_relu_conv(hc + hc, hc)
        self.merge53 = _conv_relu_conv(hc + hc, hc)
        self.merge32 = _conv_relu_conv(hc + (hc // 2), hc // 2)
        self.merge21 = _conv_relu_conv((hc // 2) + (hc // 2), hc // 2)

        # Head -> produce 2-channel logits (region + affinity)
        self.head = nn.Sequential(
            nn.Conv2d(hc // 2, hc // 4 if hc >= 64 else hc // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc // 4 if hc >= 64 else hc // 2, 2, kernel_size=1)  # 2 channels -> region, affinity
        )

        # Freeze logic
        if feature_extract:
            # freeze all backbone parameters
            for p in self.stage1.parameters():
                p.requires_grad = False
            for p in self.stage2.parameters():
                p.requires_grad = False
            for p in self.stage3.parameters():
                p.requires_grad = False
            for p in self.stage4.parameters():
                p.requires_grad = False
            for p in self.stage5.parameters():
                p.requires_grad = False
        else:
            # Partial freeze using freeze_until like conv2_2 -> freeze stage1 & stage2
            if freeze_until and isinstance(freeze_until, str) and freeze_until.lower() not in ("none", ""):
                block_num = self._parse_conv_block_index(freeze_until)
                if block_num is not None:
                    stages = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]
                    # freeze full stages with index < block_num
                    # conv1 -> block_num=1 => freeze none; conv2 -> block_num=2 => freeze stage1
                    # we freeze stages index < block_num (1-based convN)
                    for i in range(min(block_num - 1, len(stages))):
                        for p in stages[i].parameters():
                            p.requires_grad = False

        # Logging printout: how many parameters are trainable vs total
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[CraftNet Info:] \nBackbone: {backbone}, \npretrained={pretrained}, \nfeature_extract={feature_extract}")
        print(f"freeze_until='{freeze_until}' \ntrainable params: {trainable}/{total_params}\n")

    @staticmethod
    def _parse_conv_block_index(freeze_until: str) -> Optional[int]:
        """
        Parse strings like 'conv2_2' or 'conv3' and return block index integer (1-based).
        Returns None if parsing fails.
        """
        m = re.match(r"conv(\d+)", freeze_until.lower())
        if not m:
            return None
        try:
            idx = int(m.group(1))
            return idx
        except Exception:
            return None

    def _ensure_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x has 3 channels for VGG backbone.
        If model in_ch == 1, input is likely (B,1,H,W). We'll repeat channels.
        If in_ch == 3, pass through.
        Otherwise we linearly project using 1x1 conv.
        """
        if x.shape[1] == 3:
            return x
        if x.shape[1] == 1:
            # repeat grayscale to 3 channels
            return x.repeat(1, 3, 1, 1)
        # else create a small 1x1 conv to map to 3 channels (lazy create)
        # create on-the-fly conv (registered) if not exists
        if not hasattr(self, "_in_conv") or not isinstance(self._in_conv, nn.Conv2d):
            self._in_conv = nn.Conv2d(x.shape[1], 3, kernel_size=1)
            # move to same device as model
            self._in_conv = self._in_conv.to(next(self.parameters()).device)
        return self._in_conv(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass -> returns dict with keys:
          - 'region_logit' : (B,1,H_out,W_out) logits
          - 'affinity_logit': (B,1,H_out,W_out) logits
        NOTE: We upsample logits to the input spatial resolution to make loss computation simpler.
        """
        orig_device = x.device
        batch_size = x.shape[0]
        spatial_size = x.shape[2:]  # (H, W) of input

        # ensure x has 3 channels for pretrained VGG
        x_in = self._ensure_input_channels(x)

        # backbone features
        f1 = self.stage1(x_in)  # relatively high-res
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        # lateral convs
        p5 = self.conv5(f5)  # B, hc, H5, W5
        p4 = self.conv4(f4)  # B, hc, H4, W4
        # merge p5->p4
        x54 = F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode="bilinear", align_corners=False)
        x54 = torch.cat([x54, p4], dim=1)  # B, 2*hc, H4, W4
        x54 = self.merge54(x54)  # B, hc, H4, W4

        p3 = self.conv3(f3)  # B, hc, H3, W3
        x43 = F.interpolate(x54, size=(p3.shape[2], p3.shape[3]), mode="bilinear", align_corners=False)
        x43 = torch.cat([x43, p3], dim=1)  # B, 2*hc, H3, W3
        x43 = self.merge53(x43)  # B, hc, H3, W3

        p2 = self.conv2(f2)  # B, hc//2, H2, W2
        x32 = F.interpolate(x43, size=(p2.shape[2], p2.shape[3]), mode="bilinear", align_corners=False)
        x32 = torch.cat([x32, p2], dim=1)  # B, hc + hc//2 -> merge
        x32 = self.merge32(x32)  # B, hc//2, H2, W2

        p1 = self.conv1(f1)  # B, hc//2, H1, W1
        x21 = F.interpolate(x32, size=(p1.shape[2], p1.shape[3]), mode="bilinear", align_corners=False)
        x21 = torch.cat([x21, p1], dim=1)  # B, hc//2 + hc//2
        x21 = self.merge21(x21)  # B, hc//2, H1, W1

        # head -> 2 channels
        head_feat = self.head(x21)  # B, 2, H1, W1

        # upsample logits to original input spatial size
        # we upsample to x.shape[2:], which is the processed input size
        logits = F.interpolate(head_feat, size=(spatial_size[0], spatial_size[1]), mode="bilinear", align_corners=False)

        # split into region and affinity logits (each single-channel)
        region_logit = logits[:, 0:1, :, :].contiguous()
        affinity_logit = logits[:, 1:2, :, :].contiguous()

        return {"region_logit": region_logit, "affinity_logit": affinity_logit}

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
