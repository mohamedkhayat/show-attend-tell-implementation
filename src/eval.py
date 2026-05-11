import logging
from pathlib import Path

import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu

logger = logging.getLogger(__name__)


def generate_caption(model, img_tensor, device):
    """Greedy-decode one caption for a (C,H,W) image tensor."""
    model.eval()
    with torch.no_grad():
        preds, alphas = model(img_tensor.unsqueeze(0).to(device))
    word_ids = preds[0].argmax(dim=-1).cpu().tolist()
    caption = model.dec.vocab.decode(word_ids)
    attention = alphas[0].cpu()  # (n_steps, L)
    return caption, attention


def compute_bleu(model, dataset, transforms, device):
    """Compute corpus BLEU-1..4 on every unique image in *dataset*."""
    model.eval()
    vocab = model.dec.vocab
    unique_paths = list(dict.fromkeys(dataset.img_paths))
    hypotheses = []
    references_list = []

    with torch.no_grad():
        for img_path in unique_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms(img)
            preds, _ = model(img_tensor.unsqueeze(0).to(device))
            word_ids = preds[0].argmax(dim=-1).cpu().tolist()
            generated = vocab.decode(word_ids)
            refs = dataset.get_all_captions_for_image(img_path)
            hypotheses.append(generated.lower().split())
            references_list.append([r.lower().split() for r in refs])

    scores = {
        "bleu1": corpus_bleu(references_list, hypotheses, weights=(1, 0, 0, 0)),
        "bleu2": corpus_bleu(references_list, hypotheses, weights=(0.5, 0.5, 0, 0)),
        "bleu3": corpus_bleu(references_list, hypotheses, weights=(1/3, 1/3, 1/3, 0)),
        "bleu4": corpus_bleu(references_list, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)),
    }
    return scores


def evaluate_checkpoint(checkpoint_path, cfg, device=None):
    """Load a checkpoint and run full BLEU evaluation on the test split."""
    from src.models.model import Model
    from src.dataset.AnnotationDataset import AnnotationDataset
    from src.dataset.vocabulary import Vocabulary
    from src.models.transforms_factory import get_transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Support both old (cfg.model.*) and new (cfg.encoder.*) config layouts
    enc_name = cfg.encoder.name if hasattr(cfg, "encoder") else cfg.model.enc_model_name

    vocab = Vocabulary.load(str(Path(cfg.data.path) / "vocab.json"))
    _, test_transforms = get_transforms(enc_name)

    test_dataset = AnnotationDataset(
        cfg.data.path, split_type="test",
        vocab=vocab, transforms=test_transforms,
        max_length=cfg.data.max_seq_len,
    )

    model = Model(
        device=device,
        enc_model_name=enc_name,
        max_seq_len=cfg.data.max_seq_len,
        dropout_prob=0.0,
        use_tf=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(
        "Loaded %s checkpoint from epoch %d (val BLEU-4=%.4f)",
        enc_name, ckpt.get("epoch", 0), ckpt.get("bleu4", float("nan")),
    )

    scores = compute_bleu(model, test_dataset, test_transforms, device)

    print(f"\n── Test BLEU  [{enc_name}] ─────────────────")
    for k, v in scores.items():
        print(f"  {k.upper()}: {v * 100:.1f}")
    print("─────────────────────────────────────────\n")
    return scores
