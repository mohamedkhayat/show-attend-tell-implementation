# =============================================================================
# inference.py — Génération de légende pour une image (inférence)
#
# Rôle : à partir d'une image et du modèle entraîné, générer une légende
# mot par mot en mode greedy decoding (on choisit toujours le mot le plus probable).
#
# Différence avec l'entraînement :
#   - En entraînement  : teacher forcing — on donne le vrai token à chaque étape
#   - En inférence     : autoregressive — on donne la prédiction précédente comme entrée
#
# Greedy decoding :
#   À chaque pas t, on prend argmax(logits) comme prochain token.
#   Plus simple que le beam search mais légèrement moins précis.
#
# Les alphas retournés permettent de visualiser l'attention (voir visualize.py).
# =============================================================================

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset.transforms_factory import get_transforms
from src.dataset.vocabulary import Vocabulary
from src.model.encoder import Encoder
from src.model.decoder import Decoder

# Hyperparamètres — doivent correspondre exactement à ceux utilisés pendant l'entraînement
ENCODER_DIM   = 512
EMBED_DIM     = 512
DECODER_DIM   = 512
ATTENTION_DIM = 512
DROPOUT       = 0.0   # désactivé en inférence (pas de dropout au test)
MAX_LEN       = 40    # nombre max de tokens à générer avant de forcer l'arrêt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, vocab: Vocabulary):
    """Charge encoder + decoder depuis un checkpoint et les passe en mode eval."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    encoder = Encoder(encoded_dim=ENCODER_DIM, fine_tune=False).to(DEVICE)
    decoder = Decoder(
        vocab_size=len(vocab),
        encoder_dim=ENCODER_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    # eval() désactive le dropout et le batch norm en mode inférence
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def greedy_caption(encoder, decoder, image_tensor: torch.Tensor, vocab: Vocabulary):
    """Génère une légende via greedy decoding (un mot à la fois).

    Contrairement au forward du Decoder (teacher forcing), ici chaque token
    prédit devient l'entrée du pas suivant — c'est le mode autorégressif réel.

    Args:
        image_tensor: (1, 3, 224, 224) image préprocessée sur DEVICE

    Returns:
        caption : str — la légende générée
        alphas  : (T, L) — poids d'attention pour chaque token généré (pour visualize.py)
    """
    features = encoder(image_tensor)           # (1, 196, 512) — annotation vectors
    h, c     = decoder._init_hidden(features)  # initialisation LSTM depuis mean(aᵢ)

    start_idx = vocab.word2idx["<start>"]
    end_idx   = vocab.word2idx["<end>"]

    word_idx = torch.tensor([start_idx], device=DEVICE)  # premier token : <start>
    alphas   = []
    words    = []

    for _ in range(MAX_LEN):
        embed = decoder.dropout(decoder.embedding(word_idx))  # (1, embed_dim)

        z_hat, alpha = decoder.attention(features, h)         # contexte visuel + poids
        beta  = torch.sigmoid(decoder.f_beta(h))              # gating scalar βₜ
        z_hat = beta * z_hat

        lstm_input = torch.cat([embed, z_hat], dim=1)         # (1, embed+enc)
        h, c = decoder.lstm_cell(lstm_input, (h, c))

        # Deep output : même formule que dans decoder.py
        out = decoder.L_o(
            decoder.dropout(embed + decoder.L_h(h) + decoder.L_z(z_hat))
        )  # (1, vocab_size)

        word_idx = out.argmax(dim=1)   # greedy : on prend le mot le plus probable
        token    = int(word_idx.item())

        if token == end_idx:           # arrêt dès que <end> est prédit
            break

        alphas.append(alpha.squeeze(0))                       # (196,) — une carte par mot
        words.append(vocab.idx2word.get(token, "<unk>"))

    caption = " ".join(words)
    alphas  = torch.stack(alphas, dim=0) if alphas else torch.zeros(1, features.size(1))
    # alphas shape finale : (T, 196) où T = nombre de mots générés

    return caption, alphas


def caption_image(image_path: str, checkpoint_path: str, vocab: Vocabulary) -> tuple[str, torch.Tensor]:
    """Pipeline complet : image → préprocessing → encodage → décodage → légende.

    Returns:
        caption : str          — légende générée
        alphas  : (T, L) tensor — poids d'attention (utilisés dans visualize.py)
    """
    _, val_tf = get_transforms("vgg19")

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_tf(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    encoder, decoder = load_model(checkpoint_path, vocab)

    caption, alphas = greedy_caption(encoder, decoder, image_tensor, vocab)
    return caption, alphas
