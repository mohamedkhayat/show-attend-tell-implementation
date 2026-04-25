import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.AnnotationDataset import AnnotationDataset
from src.dataset.transforms_factory import get_transforms
from src.model.encoder import Encoder
from src.model.decoder import Decoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH     = "data/flicker8k"
CHECKPOINT_DIR = Path("checkpoints")

ENCODER_DIM   = 512
EMBED_DIM     = 512
DECODER_DIM   = 512
ATTENTION_DIM = 512
DROPOUT       = 0.5

BATCH_SIZE    = 16      # 64 dans le papier — réduit pour CPU/GPU faible
EPOCHS        = 20
LR            = 4e-4
LAMBDA_DS     = 1.0     # doubly stochastic regularization weight
GRAD_CLIP     = 5.0     # gradient clipping max norm
FINE_TUNE_AFTER = 999   # epoch à partir de laquelle on fine-tune l'encodeur
                        # (999 = jamais par défaut — nécessite GPU puissant)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(encoder, decoder, optimizer_dec, epoch, best_loss, path):
    torch.save({
        "epoch":       epoch,
        "best_loss":   best_loss,
        "encoder":     encoder.state_dict(),
        "decoder":     decoder.state_dict(),
        "optimizer":   optimizer_dec.state_dict(),
    }, path)


def load_checkpoint(path, encoder, decoder, optimizer_dec):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer_dec.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_loss"]


def compute_loss(logits, captions, alphas, pad_idx, lambda_ds):
    """Cross-entropy + doubly stochastic regularization (Eq. 9 of the paper).

    logits  : (B, T-1, vocab_size)
    captions: (B, T)   — targets are tokens 1..T-1
    alphas  : (B, T-1, L)
    """
    B, T_minus1, V = logits.shape

    # Targets = tokens at positions 1..T-1
    targets = captions[:, 1:]                           # (B, T-1)

    loss_ce = nn.CrossEntropyLoss(ignore_index=pad_idx)(
        logits.reshape(B * T_minus1, V),
        targets.reshape(B * T_minus1),
    )

    # Doubly stochastic: encourage Σ_t α_ti ≈ 1 for every location i
    # alphas : (B, T-1, L) → sum over time → (B, L)
    reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()

    return loss_ce + lambda_ds * reg, loss_ce.item(), reg.item()


# ---------------------------------------------------------------------------
# Train / Val one epoch
# ---------------------------------------------------------------------------

def run_epoch(encoder, decoder, loader, optimizer_dec, pad_idx, train: bool):
    encoder.train(train)
    decoder.train(train)

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for images, captions in loader:
            images   = images.to(DEVICE)
            captions = captions.to(DEVICE)

            features           = encoder(images)
            logits, alphas     = decoder(features, captions)

            loss, ce, reg = compute_loss(
                logits, captions, alphas, pad_idx, LAMBDA_DS
            )

            if train:
                optimizer_dec.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
                optimizer_dec.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # --- Data ---
    train_tf, val_tf = get_transforms("vgg19")

    train_ds = AnnotationDataset(DATA_PATH, split_type="train", transforms=train_tf)
    val_ds   = AnnotationDataset(DATA_PATH, split_type="val",   transforms=val_tf,
                                  vocab=train_ds.vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")

    pad_idx   = train_ds.vocab.word2idx["<pad>"]
    vocab_size = len(train_ds.vocab)

    print(f"Device     : {DEVICE}")
    print(f"Vocab size : {vocab_size}")
    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")

    # --- Models ---
    encoder = Encoder(encoded_dim=ENCODER_DIM, fine_tune=False).to(DEVICE)
    decoder = Decoder(
        vocab_size=vocab_size,
        encoder_dim=ENCODER_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Only decoder parameters are optimised at first (encoder is frozen)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path     = CHECKPOINT_DIR / "best.pt"

    # --- Loop ---
    for epoch in range(1, EPOCHS + 1):
        # Optionally unfreeze encoder after FINE_TUNE_AFTER epochs
        if epoch == FINE_TUNE_AFTER:
            encoder.set_fine_tune(True)
            print(f"[Epoch {epoch}] Fine-tuning encoder enabled.")

        t0 = time.time()
        train_loss = run_epoch(encoder, decoder, train_loader, optimizer_dec,
                               pad_idx, train=True)
        val_loss   = run_epoch(encoder, decoder, val_loader,   optimizer_dec,
                               pad_idx, train=False)
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(encoder, decoder, optimizer_dec, epoch,
                            best_val_loss, best_path)
            print(f"  → checkpoint saved (val {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
