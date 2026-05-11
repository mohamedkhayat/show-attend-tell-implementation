import logging

import torch
import torch.nn as nn
import torchvision.models as tvm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: name → (output_dim, builder)
# All builders return a module whose forward: (B,3,224,224) → (B, L, D)
# ---------------------------------------------------------------------------

def _build_vgg19():
    backbone = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT)
    # features[:29] = conv1-conv4 up to (but not including) the final maxpool
    # Output: (B, 512, 14, 14) for 224×224 input
    return backbone.features[:29]


def _build_resnet(name):
    weights_cls = {
        "resnet50": tvm.ResNet50_Weights.DEFAULT,
        "resnet101": tvm.ResNet101_Weights.DEFAULT,
    }[name]
    model_fn = {"resnet50": tvm.resnet50, "resnet101": tvm.resnet101}[name]
    backbone = model_fn(weights=weights_cls)
    # After layer3: (B, 1024, 14, 14) for 224×224 input — matches the paper's 14×14 grid
    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )


def _build_clip(hf_id="openai/clip-vit-base-patch16"):
    from transformers import CLIPVisionModel
    return CLIPVisionModel.from_pretrained(hf_id)


ENCODER_REGISTRY = {
    "vgg19":      {"dim": 512,  "type": "cnn",  "build": _build_vgg19},
    "resnet50":   {"dim": 1024, "type": "cnn",  "build": lambda: _build_resnet("resnet50")},
    "resnet101":  {"dim": 1024, "type": "cnn",  "build": lambda: _build_resnet("resnet101")},
    "clip_vit_b16": {
        "dim": 768,
        "type": "clip",
        "build": lambda: _build_clip("openai/clip-vit-base-patch16"),
    },
}


class Encoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if model_name not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{model_name}'. "
                f"Available: {list(ENCODER_REGISTRY)}"
            )
        cfg = ENCODER_REGISTRY[model_name]
        self.model_name = model_name
        self.enc_type = cfg["type"]
        self.output_dim = cfg["dim"]  # read by Model to configure decoder

        self.backbone = cfg["build"]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, L, D)  where L=196, D=output_dim"""
        if self.enc_type == "cnn":
            # VGG19 → (B,512,14,14) | ResNet → (B,1024,14,14)
            feats = self.backbone(x)                         # (B, D, H, W)
            B, D, H, W = feats.shape
            out = feats.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, L, D)
        else:  # clip
            out = self.backbone(pixel_values=x).last_hidden_state  # (B, L+1, D)
            out = out[:, 1:, :]                              # drop CLS → (B, 196, D)

        logger.debug("Encoder %s → %s", self.model_name, tuple(out.shape))
        return out
