"""
CLI entry point for Show, Attend and Tell.

Usage
-----
Train (default encoder: vgg19):
  python -m src.main

Train with a different encoder:
  python -m src.main encoder=resnet50
  python -m src.main encoder=clip_vit_b16

Evaluate a checkpoint:
  python -m src.main mode=eval
  python -m src.main mode=eval encoder=resnet50

Generate a caption for one image:
  python -m src.main mode=caption caption.image_path=data/flicker8k/Images/xxx.jpg

Override any config key on the fly:
  python -m src.main encoder=vgg19 data.batch_size=64 training.lr=1e-3
"""

import logging

import hydra
from omegaconf import DictConfig

from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging("INFO")

    mode = cfg.get("mode", "train")

    if mode == "train":
        from src.train import train
        train(cfg)

    elif mode == "eval":
        import torch
        from src.eval import evaluate_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = cfg.eval.checkpoint
        evaluate_checkpoint(ckpt_path, cfg, device=device)

    elif mode == "caption":
        import torch
        import numpy as np
        from PIL import Image
        from src.models.model import Model
        from src.models.transforms_factory import get_transforms
        from src.dataset.vocabulary import Vocabulary
        from pathlib import Path

        img_path = cfg.caption.image_path
        if not img_path:
            raise ValueError("Provide caption.image_path=<path/to/image.jpg>")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab = Vocabulary.load(str(Path(cfg.data.path) / "vocab.json"))
        _, test_tf = get_transforms(cfg.encoder.name)

        model = Model(
            device=device,
            enc_model_name=cfg.encoder.name,
            max_seq_len=cfg.data.max_seq_len,
            dropout_prob=0.0,
            use_tf=False,
        ).to(device)

        import torch as _torch
        ckpt = _torch.load(cfg.eval.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        img = Image.open(img_path).convert("RGB")
        img_tensor = test_tf(img).unsqueeze(0).to(device)

        with _torch.no_grad():
            preds, alphas = model(img_tensor)
        word_ids = preds[0].argmax(dim=-1).cpu().tolist()
        caption = vocab.decode(word_ids)
        print(f"\nImage : {img_path}")
        print(f"Caption: {caption}\n")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use mode=train / mode=eval / mode=caption")


if __name__ == "__main__":
    main()
