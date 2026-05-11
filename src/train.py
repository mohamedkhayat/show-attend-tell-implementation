import logging
import os
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.AnnotationDataset import AnnotationDataset
from src.dataset.vocabulary import Vocabulary
from src.eval import compute_bleu
from src.models.model import Model
from src.models.transforms_factory import get_transforms
from src.utils.data import caption_collate_fn
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _build_loaders(cfg, vocab, train_tf, test_tf):
    collate = partial(caption_collate_fn, pad_idx=vocab.word2idx["<pad>"])
    train_ds = AnnotationDataset(
        cfg.data.path, split_type="train",
        vocab=vocab, transforms=train_tf,
        max_length=cfg.data.max_seq_len,
    )
    val_ds = AnnotationDataset(
        cfg.data.path, split_type="val",
        vocab=vocab, transforms=test_tf,
        max_length=cfg.data.max_seq_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
        collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        collate_fn=collate, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


def _train_epoch(model, loader, optimizer, scaler, device, cfg, amp_dtype):
    model.train()
    pad_idx = model.dec.vocab.word2idx["<pad>"]
    total_loss = total_ce = total_reg = 0.0

    for images, captions in tqdm(loader, desc="  train", leave=False):
        images  = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=cfg.training.mixed_precision):
            preds, alphas = model(images, captions)
            # preds:  (B, T-1, V)   captions[:, 1:]: (B, T-1)
            V = preds.size(2)
            loss_ce = F.cross_entropy(
                preds.reshape(-1, V),
                captions[:, 1:].reshape(-1),
                ignore_index=pad_idx,
            )
            reg = cfg.training.lambda_reg * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss = loss_ce + reg

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.dec.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dec.parameters(), cfg.training.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_ce   += loss_ce.item()
        total_reg  += reg.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_reg / n


def train(cfg: DictConfig) -> None:
    configure_logging("INFO")
    logger.info("Encoder: %s  (dim=%d)", cfg.encoder.name, cfg.encoder.dim)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    logger.info("Device: %s", device)

    # Mixed precision — bf16 preferred on RTX 5090 (Blackwell), no GradScaler needed
    use_mp = cfg.training.mixed_precision and device.type == "cuda"
    bf16_ok = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    scaler = None if (not use_mp or bf16_ok) else GradScaler()
    if use_mp:
        logger.info("Mixed precision: %s (GradScaler=%s)", amp_dtype, scaler is not None)

    vocab = Vocabulary.load(str(Path(cfg.data.path) / "vocab.json"))
    train_tf, test_tf = get_transforms(cfg.encoder.name)

    train_loader, val_loader, train_ds, val_ds = _build_loaders(
        cfg, vocab, train_tf, test_tf
    )
    logger.info("Data: %d train / %d val", len(train_ds), len(val_ds))

    model = Model(
        device=device,
        enc_model_name=cfg.encoder.name,
        max_seq_len=cfg.data.max_seq_len,
        dropout_prob=cfg.decoder.dropout_prob,
        use_tf=cfg.decoder.use_tf,
    ).to(device)

    # Freeze encoder — paper uses fixed CNN features
    for param in model.enc.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen; training decoder only.")

    if cfg.training.compile:
        model.dec = torch.compile(model.dec, dynamic=True)
        logger.info("torch.compile enabled on decoder.")

    optimizer = torch.optim.RMSprop(
        model.dec.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity") or None,
        mode=cfg.wandb.get("mode", "offline"),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.encoder.name,
        reinit=True,
    )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_bleu4 = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.training.epochs + 1):
        avg_loss, avg_ce, avg_reg = _train_epoch(
            model, train_loader, optimizer, scaler, device, cfg, amp_dtype
        )

        log_dict = {
            "epoch": epoch,
            "train/loss": avg_loss,
            "train/ce_loss": avg_ce,
            "train/reg_loss": avg_reg,
        }

        if epoch % cfg.training.bleu_eval_freq == 0:
            bleu = compute_bleu(model, val_ds, test_tf, device)
            bleu4 = bleu["bleu4"]
            log_dict.update({f"val/{k}": v for k, v in bleu.items()})
            bleu_str = "  ".join(f"{k.upper()}={v*100:.2f}" for k, v in bleu.items())
            logger.info(
                "Epoch %d/%d | loss=%.4f  ce=%.4f  reg=%.4f | %s",
                epoch, cfg.training.epochs, avg_loss, avg_ce, avg_reg, bleu_str,
            )

            if bleu4 > best_bleu4:
                best_bleu4 = bleu4
                patience_counter = 0
                ckpt_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "bleu4": best_bleu4,
                        "encoder": cfg.encoder.name,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )
                logger.info("  ✓ New best BLEU-4=%.4f  saved → %s", best_bleu4, ckpt_path)
            else:
                patience_counter += 1
                logger.info(
                    "  No improvement (%d/%d patience)", patience_counter, cfg.training.patience
                )
                if patience_counter >= cfg.training.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    wandb.log(log_dict)
                    break
        else:
            logger.info(
                "Epoch %d/%d | loss=%.4f  ce=%.4f  reg=%.4f",
                epoch, cfg.training.epochs, avg_loss, avg_ce, avg_reg,
            )

        wandb.log(log_dict)

    wandb.finish()
    logger.info("Training complete. Best val BLEU-4=%.4f", best_bleu4)
