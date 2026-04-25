import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.AnnotationDataset import AnnotationDataset
from src.dataset.transforms_factory import get_transforms
from src.model.encoder import Encoder
from src.model.decoder import Decoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH      = "data/flicker8k"
CHECKPOINT_DIR = Path("checkpoints")

ENCODER_DIM   = 512
EMBED_DIM     = 512
DECODER_DIM   = 512
ATTENTION_DIM = 512
DROPOUT       = 0.5

BATCH_SIZE      = 32       # T4 has 16 GB VRAM — safe to double from 16
EPOCHS          = 30       # early stopping will interrupt well before
LR              = 4e-4
LAMBDA_DS       = 1.0      # doubly stochastic regularization weight
GRAD_CLIP       = 5.0      # gradient clipping max norm
PATIENCE        = 5        # early stopping: epochs without val improvement
LR_PATIENCE     = 2        # reduce LR after this many epochs without improvement
LR_FACTOR       = 0.5      # multiply LR by this factor on plateau
FINE_TUNE_AFTER = 999      # epoch to unfreeze encoder (999 = never by default)
NUM_WORKERS     = 2        # Colab has 2 CPU cores available for data loading

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, best_loss, path):
    torch.save({
        "epoch":      epoch,
        "best_loss":  best_loss,
        "encoder":    encoder.state_dict(),
        "decoder":    decoder.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }, path)


def load_checkpoint(path, encoder, decoder, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_loss"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(logits, captions, alphas, pad_idx, lambda_ds):
    """Cross-entropy + doubly stochastic regularization (Eq. 9 of the paper).

    logits  : (B, T-1, vocab_size)
    captions: (B, T)
    alphas  : (B, T-1, L)
    """
    B, T_minus1, V = logits.shape
    targets = captions[:, 1:]                            # (B, T-1)

    loss_ce = nn.CrossEntropyLoss(ignore_index=pad_idx)(
        logits.reshape(B * T_minus1, V),
        targets.reshape(B * T_minus1),
    )

    reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()

    return loss_ce + lambda_ds * reg, loss_ce.item(), reg.item()


# ---------------------------------------------------------------------------
# Train / Val one epoch
# ---------------------------------------------------------------------------

def run_epoch(encoder, decoder, loader, optimizer, pad_idx, train: bool):
    encoder.train(train)
    decoder.train(train)

    total_loss = 0.0
    n_batches  = 0
    ctx        = torch.enable_grad() if train else torch.no_grad()
    desc       = "train" if train else "val  "

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for images, captions in pbar:
            images   = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)

            features       = encoder(images)
            logits, alphas = decoder(features, captions)

            loss, ce, reg = compute_loss(logits, captions, alphas, pad_idx, LAMBDA_DS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce:.4f}", reg=f"{reg:.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / "best.pt"

    # --- Data ---
    train_tf, val_tf = get_transforms("vgg19")

    train_ds = AnnotationDataset(DATA_PATH, split_type="train", transforms=train_tf)
    val_ds   = AnnotationDataset(DATA_PATH, split_type="val",   transforms=val_tf,
                                 vocab=train_ds.vocab)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    pad_idx    = train_ds.vocab.word2idx["<pad>"]
    vocab_size = len(train_ds.vocab)

    print(f"Device     : {DEVICE}")
    print(f"Vocab size : {vocab_size}")
    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Batch size : {BATCH_SIZE}  |  Batches/epoch : {len(train_loader)}")

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

    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # Resume from checkpoint if one exists
    start_epoch   = 1
    best_val_loss = float("inf")
    if best_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            best_path, encoder, decoder, optimizer, scheduler
        )
        start_epoch += 1
        print(f"Resumed from checkpoint — starting at epoch {start_epoch}, best val {best_val_loss:.4f}")

    # --- Training loop ---
    epochs_no_improve = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == FINE_TUNE_AFTER:
            encoder.set_fine_tune(True)
            for param_group in optimizer.param_groups:
                param_group["lr"] = LR / 10
            print(f"[Epoch {epoch}] Fine-tuning encoder enabled (LR → {LR/10:.1e})")

        t0         = time.time()
        train_loss = run_epoch(encoder, decoder, train_loader, optimizer, pad_idx, train=True)
        val_loss   = run_epoch(encoder, decoder, val_loader,   optimizer, pad_idx, train=False)
        elapsed    = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.1e} | {elapsed:.0f}s"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, best_val_loss, best_path)
            print(f"  → checkpoint saved (val {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  → no improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
