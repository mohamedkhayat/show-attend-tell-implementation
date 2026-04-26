# =============================================================================
# evaluate.py — Évaluation BLEU sur le split de validation
#
# Rôle : mesurer quantitativement la qualité des légendes générées par rapport
# aux 5 légendes humaines de référence disponibles pour chaque image.
#
# Métrique BLEU (Bilingual Evaluation Understudy) :
#   Compare les n-grammes (séquences de n mots) entre la prédiction et les
#   références. BLEU-1 compare les mots isolés, BLEU-4 les séquences de 4 mots.
#   Un score élevé indique que les mots et leur ordre correspondent aux références.
#
# Résultats obtenus sur Flickr8k (val set, 809 images uniques) :
#   BLEU-1 : 61.66%   (papier original : 67.0%)
#   BLEU-2 : 40.67%   (papier original : 45.7%)
#   BLEU-3 : 27.09%   (papier original : 31.4%)
#   BLEU-4 : 17.99%   (papier original : 21.3%)
#
# Écart expliqué par : pas de fine-tuning encodeur, batch 32 vs 64, early stopping
# =============================================================================

import os
from collections import defaultdict

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from src.dataset.AnnotationDataset import AnnotationDataset
from src.dataset.transforms_factory import get_transforms
from src.dataset.vocabulary import Vocabulary
from src.inference import load_model, greedy_caption

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/flicker8k"


def evaluate(checkpoint_path: str, vocab_path: str):
    """Calcule les scores BLEU-1/2/3/4 sur le val set complet.

    Stratégie :
      1. Grouper les 5 références par image unique (un dict img_path → liste de refs)
      2. Pour chaque image unique, générer une prédiction via greedy decoding
      3. Calculer corpus_bleu sur l'ensemble des paires (refs, hypothèse)

    Args:
        checkpoint_path : chemin vers best.pt
        vocab_path      : chemin vers vocab.json
    """
    vocab = Vocabulary.load(vocab_path)

    _, val_tf = get_transforms("vgg19")
    val_ds = AnnotationDataset(DATA_PATH, split_type="val", transforms=val_tf, vocab=vocab)

    encoder, decoder = load_model(checkpoint_path, vocab)

    # Regroupe les 5 références humaines par image (clé = chemin de l'image)
    refs_by_image = defaultdict(list)
    for img_path, caption in zip(val_ds.img_paths, val_ds.captions):
        refs_by_image[img_path].append(caption.lower().split())  # tokenisé en liste de mots

    unique_images = list(refs_by_image.keys())  # 809 images uniques dans Flickr8k val

    hypotheses = []  # liste des prédictions tokenisées
    references = []  # liste des listes de références (5 par image)

    for img_path in tqdm(unique_images, desc="evaluating"):
        image = __import__("PIL").Image.open(img_path).convert("RGB")
        image_tensor = val_tf(image).unsqueeze(0).to(DEVICE)

        caption, _ = greedy_caption(encoder, decoder, image_tensor, vocab)
        hyp = caption.lower().split()

        hypotheses.append(hyp)
        references.append(refs_by_image[img_path])  # les 5 références pour cette image

    # SmoothingFunction.method1 : évite les scores BLEU nuls pour les n-grammes rares
    smoother = SmoothingFunction().method1

    # weights définissent les poids des n-grammes : BLEU-1 → unigrams seulement, etc.
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),              smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),          smoothing_function=smoother)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0),        smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),  smoothing_function=smoother)

    print(f"\n{'='*40}")
    print(f"  BLEU-1 : {bleu1*100:.2f}%")
    print(f"  BLEU-2 : {bleu2*100:.2f}%")
    print(f"  BLEU-3 : {bleu3*100:.2f}%")
    print(f"  BLEU-4 : {bleu4*100:.2f}%")
    print(f"{'='*40}")
    print(f"  Images evaluated : {len(unique_images)}")

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4}


if __name__ == "__main__":
    evaluate(
        checkpoint_path="checkpoints/best.pt",
        vocab_path="checkpoints/vocab.json",
    )
