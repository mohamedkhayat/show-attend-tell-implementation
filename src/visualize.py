# =============================================================================
# visualize.py — Visualisation de l'attention sur l'image
#
# Rôle : montrer visuellement QUELLES régions de l'image le modèle regardait
# pour générer chaque mot de la légende — reproduit la Figure 3 du papier.
#
# Principe :
#   Pour chaque mot généré, on récupère le vecteur αₜ de taille 196 (14×14).
#   On le redimensionne à la taille de l'image originale via interpolation
#   bilinéaire (zoom ×16), puis on le superpose en heatmap colorée (colormap jet).
#
# Interprétation des couleurs :
#   Rouge/jaune = forte attention (le modèle regarde cette zone)
#   Bleu/violet = faible attention (le modèle ignore cette zone)
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

from src.dataset.vocabulary import Vocabulary
from src.inference import caption_image, load_model, greedy_caption
from src.dataset.transforms_factory import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_attention(
    image_path: str,
    checkpoint_path: str,
    vocab: Vocabulary,
    save_path: str | None = None,
):
    """Génère une grille d'images avec heatmap d'attention par mot.

    Args:
        image_path      : chemin vers l'image d'entrée
        checkpoint_path : chemin vers best.pt
        vocab           : vocabulaire chargé
        save_path       : si fourni, sauvegarde la figure au lieu de l'afficher
    """
    # --- Inférence ---
    _, val_tf = get_transforms("vgg19")
    image_orig   = Image.open(image_path).convert("RGB")
    image_tensor = val_tf(image_orig).unsqueeze(0).to(DEVICE)

    encoder, decoder = load_model(checkpoint_path, vocab)
    caption, alphas  = greedy_caption(encoder, decoder, image_tensor, vocab)

    words = caption.split()
    T     = len(words)

    # Reshape : (T, 196) → (T, 14, 14) pour pouvoir zoomer chaque carte
    alphas_np = alphas.cpu().numpy().reshape(T, 14, 14)

    # --- Mise en page ---
    cols = 4
    rows = (T + cols - 1) // cols + 1  # +1 pour la ligne avec l'image originale

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # Premier panneau : image originale + légende complète
    axes[0].imshow(image_orig)
    axes[0].set_title(f'"{caption}"', fontsize=8, wrap=True)
    axes[0].axis("off")

    # Un panneau par mot avec heatmap d'attention superposée
    for i, (word, alpha) in enumerate(zip(words, alphas_np)):
        ax = axes[i + 1]

        # Upsample la carte 14×14 à la taille de l'image originale (interpolation bilinéaire)
        h, w     = image_orig.size[1], image_orig.size[0]
        alpha_up = zoom(alpha, (h / 14, w / 14), order=1)  # order=1 = bilinéaire

        ax.imshow(image_orig)
        # alpha=0.45 : semi-transparence pour voir l'image sous la heatmap
        ax.imshow(alpha_up, cmap="jet", alpha=0.45)
        ax.set_title(word, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Cache les panneaux inutilisés (grille pas toujours remplie entièrement)
    for j in range(T + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Show, Attend and Tell — attention visualization", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return caption, alphas
