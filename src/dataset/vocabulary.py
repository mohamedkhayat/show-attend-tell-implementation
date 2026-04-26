# =============================================================================
# vocabulary.py — Gestion du vocabulaire (mots ↔ indices)
#
# Rôle : convertir les légendes textuelles en séquences d'indices numériques
# que le LSTM peut traiter, et inversement convertir les prédictions du modèle
# en texte lisible.
#
# Tokens spéciaux réservés :
#   <pad>   (index 0) : rembourrage pour aligner les séquences dans un batch
#   <start> (index 1) : signal de début — premier token donné au décodeur
#   <end>   (index 2) : signal de fin — le décodeur s'arrête quand il le prédit
#   <unk>   (index 3) : remplace les mots trop rares (fréquence < min_freq)
#
# FIX appliqué :
#   Ajout de la méthode __len__ qui manquait → levait TypeError au démarrage :
#   "TypeError: object of type 'Vocabulary' has no len()"
#   Utilisée dans train.py : vocab_size = len(train_ds.vocab)
# =============================================================================

import json
from collections import Counter
import torch


class Vocabulary:
    def __init__(self, min_freq=5, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # Les 4 tokens spéciaux sont pré-assignés aux indices 0-3.
        # Tous les vrais mots commenceront à l'index 4.
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()

    def __len__(self):
        # FIX : méthode manquante — ajoutée pour corriger TypeError dans train.py
        # vocab_size = len(train_ds.vocab) → nécessaire pour créer nn.Embedding(vocab_size, ...)
        return len(self.word2idx)

    def build_vocab(self, captions):
        """Construit le vocabulaire depuis une liste de légendes.

        Seuls les mots apparaissant au moins min_freq fois sont inclus.
        Résultat sur Flickr8k : 2669 mots (seuil min_freq=5).
        """
        for caption in captions:
            words = caption.lower().split()
            self.word_count.update(words)

        # Filtre par fréquence minimale et limite la taille totale du vocab
        words = [
            word for word, freq in self.word_count.items() if freq >= self.min_freq
        ]
        words = words[: self.max_vocab_size - len(self.word2idx)]

        # Les vrais mots commencent à l'index 4 (après les 4 tokens spéciaux)
        for idx, word in enumerate(words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, caption, max_length=20):
        """Convertit une légende en tensor d'indices avec padding.

        Structure de sortie : [<start>, w1, w2, ..., wN, <end>, <pad>, <pad>, ...]
        Tronqué ou padé pour atteindre exactement max_length tokens.
        """
        words = caption.lower().split()
        # Les mots absents du vocabulaire sont remplacés par <unk>
        indices = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]

        indices = [self.word2idx["<start>"]] + indices + [self.word2idx["<end>"]]

        # Padding jusqu'à max_length, puis troncature si dépassement
        indices += [self.word2idx["<pad>"]] * (max_length - len(indices))
        return torch.tensor(indices[:max_length])

    def decode(self, indices):
        """Convertit une séquence d'indices en texte lisible.

        Ignore <start>, <pad> et s'arrête à <end>.
        Compatible avec torch tensors, numpy scalars et int Python.
        """
        words = []
        start_idx = self.word2idx["<start>"]
        end_idx   = self.word2idx["<end>"]
        pad_idx   = self.word2idx["<pad>"]

        for idx in indices:
            # Support des types torch.Tensor, numpy scalar et int
            if hasattr(idx, "item"):
                idx = int(idx.item())
            else:
                idx = int(idx)

            if idx == end_idx:
                break
            if idx in (pad_idx, start_idx):
                continue

            if idx != pad_idx:
                words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def save(self, filepath):
        """Sauvegarde le vocabulaire en JSON.

        Utilisé dans train.py juste après la construction du vocab,
        pour pouvoir faire l'inférence sans avoir à recharger tout le dataset.
        """
        with open(filepath, "w") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word,
                    "word_count": dict(self.word_count),
                },
                f,
            )

    @classmethod
    def load(cls, filepath):
        """Charge un vocabulaire depuis un fichier JSON sauvegardé.

        Note : les clés de idx2word sont converties en int car JSON
        sérialise toutes les clés en strings.
        """
        vocab = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
            vocab.word2idx = data["word2idx"]
            # FIX JSON : json.load() retourne les clés comme strings → on reconvertit en int
            vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
            vocab.word_count = Counter(data["word_count"])
        return vocab
