# =============================================================================
# train.py — Boucle d'entraînement principale
#
# Rôle : entraîner l'encodeur CNN + décodeur LSTM sur Flickr8k avec :
#   - Teacher forcing : on fournit le vrai token à chaque pas (pas la prédiction)
#   - Loss = Cross-Entropy(ignore <pad>) + λ·doubly stochastic regularization
#   - LR scheduler : réduit le LR automatiquement quand la val loss stagne
#   - Early stopping : arrête si pas d'amélioration pendant PATIENCE epochs
#   - Checkpointing : sauvegarde automatiquement le meilleur modèle (val loss)
#   - Reprise automatique : si best.pt existe, reprend depuis le dernier checkpoint
# =============================================================================

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
# Hyperparamètres
# ---------------------------------------------------------------------------

DATA_PATH      = "data/flicker8k"
CHECKPOINT_DIR = Path("checkpoints")

ENCODER_DIM   = 512   # dimension des annotation vectors VGG19
EMBED_DIM     = 512   # dimension des embeddings de mots
DECODER_DIM   = 512   # dimension de l'état caché LSTM
ATTENTION_DIM = 512   # dimension intermédiaire du MLP d'attention
DROPOUT       = 0.5   # régularisation du décodeur (désactivé en inférence)

BATCH_SIZE      = 64      # Batch size augmenté pour meilleure stabilité (comme le papier)
EPOCHS          = 30      # l'early stopping arrête avant si convergence
LR              = 4e-4    # taux d'apprentissage initial (Adam)
LAMBDA_DS       = 1.0     # poids de la doubly stochastic regularization (Eq. 9)
GRAD_CLIP       = 5.0     # norme max du gradient (évite les explosions)
PATIENCE        = 10      # early stopping : epochs sans amélioration avant arrêt (augmenté de 5 → 10)
LR_PATIENCE     = 5       # scheduler : réduit LR après N epochs sans amélioration (augmenté de 2 → 5)
LR_FACTOR       = 0.5     # facteur de réduction du LR (nouveau_LR = LR * 0.5)
FINE_TUNE_AFTER = 7       # epoch à partir de laquelle on débloque l'encodeur CNN (changé de 999 → 7)
NUM_WORKERS     = 2       # workers DataLoader (Colab a 2 CPU cores)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Sauvegarde / chargement de checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, best_loss, path):
    # On sauvegarde aussi le scheduler pour reprendre avec le bon LR
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
# Calcul de la loss
# ---------------------------------------------------------------------------

def compute_loss(logits, captions, alphas, pad_idx, lambda_ds):
    """Cross-entropy + doubly stochastic regularization (Eq. 9 du papier).

    La régularisation force le modèle à regarder chaque région au moins une
    fois au total sur la séquence : Σₜ αₜᵢ ≈ 1 pour tout i.

    Args:
        logits  : (B, T-1, vocab_size) — scores prédits
        captions: (B, T)              — tokens ground-truth
        alphas  : (B, T-1, L)         — poids d'attention à chaque étape
    """
    B, T_minus1, V = logits.shape

    # Les cibles sont les tokens à prédire : positions 1..T-1
    targets = captions[:, 1:]  # (B, T-1) — on exclut <start>

    # ignore_index=pad_idx : les positions <pad> ne contribuent pas à la loss
    loss_ce = nn.CrossEntropyLoss(ignore_index=pad_idx)(
        logits.reshape(B * T_minus1, V),
        targets.reshape(B * T_minus1),
    )

    # Doubly stochastic : somme des poids d'attention sur le temps → (B, L)
    # On veut que cette somme soit proche de 1 pour chaque région i
    reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()

    return loss_ce + lambda_ds * reg, loss_ce.item(), reg.item()


# ---------------------------------------------------------------------------
# Epoch train ou val
# ---------------------------------------------------------------------------

def run_epoch(encoder, decoder, loader, optimizer, pad_idx, train: bool):
    """Exécute une epoch complète d'entraînement ou de validation.

    Utilise torch.enable_grad() en train et torch.no_grad() en val pour
    éviter de calculer des gradients inutilement pendant la validation.
    """
    encoder.train(train)
    decoder.train(train)

    total_loss = 0.0
    n_batches  = 0
    ctx        = torch.enable_grad() if train else torch.no_grad()
    desc       = "train" if train else "val  "

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for images, captions in pbar:
            # non_blocking=True : transfert CPU→GPU asynchrone (gain de perf)
            images   = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)

            features       = encoder(images)               # (B, 196, 512)
            logits, alphas = decoder(features, captions)   # (B, T-1, vocab_size), (B, T-1, 196)

            loss, ce, reg = compute_loss(logits, captions, alphas, pad_idx, LAMBDA_DS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Clip le gradient pour éviter les explosions (problème courant avec LSTM)
                nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce:.4f}", reg=f"{reg:.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Fonction principale d'entraînement
# ---------------------------------------------------------------------------

def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_path = CHECKPOINT_DIR / "best.pt"

    # --- Données ---
    train_tf, val_tf = get_transforms("vgg19")

    train_ds = AnnotationDataset(DATA_PATH, split_type="train", transforms=train_tf)
    val_ds   = AnnotationDataset(DATA_PATH, split_type="val",   transforms=val_tf,
                                 vocab=train_ds.vocab)  # vocab partagé train→val

    # On sauvegarde le vocab dès le départ pour pouvoir faire l'inférence sans réentraîner
    train_ds.vocab.save(CHECKPOINT_DIR / "vocab.json")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),  # garde les workers entre les epochs
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

    # --- Modèles ---
    encoder = Encoder(encoded_dim=ENCODER_DIM, fine_tune=False).to(DEVICE)
    decoder = Decoder(
        vocab_size=vocab_size,
        encoder_dim=ENCODER_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Crée deux groupes de paramètres dans l'optimiseur :
    # - Encodeur : gelé au départ, LR réduit pour le fine-tuning
    # - Décodeur : entraîné depuis le départ
    optimizer = torch.optim.Adam([
        {"params": encoder.features.parameters(), "lr": LR / 10},  # Fine-tuning LR
        {"params": decoder.parameters(), "lr": LR},
    ])

    # Réduit le LR de moitié si la val loss ne s'améliore pas pendant LR_PATIENCE epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # --- Reprise depuis checkpoint si disponible ---
    start_epoch   = 1
    best_val_loss = float("inf")
    if best_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            best_path, encoder, decoder, optimizer, scheduler
        )
        start_epoch += 1
        print(f"Resumed from checkpoint — starting at epoch {start_epoch}, best val {best_val_loss:.4f}")

    # --- Boucle d'entraînement ---
    epochs_no_improve = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        # Débloque l'encodeur après FINE_TUNE_AFTER epochs (nécessite GPU puissant)
        if epoch == FINE_TUNE_AFTER and not encoder.features[0].weight.requires_grad:
            encoder.set_fine_tune(True)
            print(f"[Epoch {epoch}] Fine-tuning encoder enabled — encoder params now trainable (LR = {LR/10:.1e})")

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

        # Le scheduler ajuste le LR selon la val loss
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
