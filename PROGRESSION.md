# Progression du Projet — Show, Attend and Tell

> Implémentation du papier : *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (Xu et al., 2015)

---

## 1. Ce qui a été réalisé — étape par étape

### Étape 1 — Infrastructure du projet

**Fichiers :** `pyproject.toml`, `Dockerfile`, `requirements.txt`

- Package Python structuré en `src/` layout installable avec `pip install -e .`
- Conteneur Docker GPU-ready (base `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`)
- Dépendances déclarées : `torch`, `torchvision`, `pillow`, `pandas`, `numpy`, `wandb`

---

### Étape 2 — Gestion du vocabulaire

**Fichier :** [`src/dataset/vocabulary.py`](src/dataset/vocabulary.py)

Le vocabulaire est la brique fondamentale qui traduit les mots en indices numériques (et inversement), indispensable pour entraîner le décodeur LSTM.

**Ce qui est implémenté :**
- Tokens spéciaux : `<pad>`, `<start>`, `<end>`, `<unk>`
- `build_vocab(captions)` — construit le vocabulaire depuis les légendes avec un seuil de fréquence minimum (défaut : 5) et une taille max (défaut : 10 000 mots)
- `encode(caption)` — convertit une phrase en liste d'indices entiers avec padding
- `decode(indices)` — convertit des indices en texte lisible
- `save() / load()` — sérialisation JSON pour réutilisation entre sessions

**Exemple de flux :**
```
"a dog on a beach"
    → [1, 4, 87, 12, 4, 203, 2]   (encode)
    → "<start> a dog on a beach <end>"  (decode)
```

---

### Étape 3 — Préparation des données Flickr8k

**Fichier :** [`src/utils/data.py`](src/utils/data.py)

Le dataset Flickr8k contient 8 000 images, chacune annotée avec 5 légendes humaines. Cette étape parse les fichiers bruts et génère des splits reproductibles.

**Ce qui est implémenté :**
- `prepare_flicker_data()` — lit `captions.txt`, groupe les légendes par image, applique une division 80/10/10 (train/val/test), sauvegarde 6 fichiers JSON :
  - `train_img_paths.json`, `train_captions.json`
  - `val_img_paths.json`, `val_captions.json`
  - `test_img_paths.json`, `test_captions.json`
- `ensure_flicker_splits()` — idempotent : ne régénère que si les fichiers manquent
- Stub `prepare_coco_data()` — non implémenté (lève `NotImplementedError`)

---

### Étape 4 — Pipeline de transformations d'images

**Fichiers :** [`src/dataset/transforms_factory.py`](src/dataset/transforms_factory.py), [`src/dataset/model_factory.py`](src/dataset/model_factory.py)

Le papier utilise VGG19 pré-entraîné pour extraire des feature maps spatiales (14×14×512). Les images doivent être normalisées selon les statistiques ImageNet.

**Ce qui est implémenté :**
- `get_transforms(split, model_name)` — retourne un pipeline `torchvision.transforms.v2` :
  - **Train** : flip horizontal aléatoire + color jitter + grayscale aléatoire + normalisation VGG19
  - **Val/Test** : normalisation VGG19 uniquement
- `MODEL_WEIGHTS_MAPPING` — dict `{"vgg19": VGG19_Weights}` (extensible à ResNet, etc.)

---

### Étape 5 — Dataset PyTorch

**Fichier :** [`src/dataset/AnnotationDataset.py`](src/dataset/AnnotationDataset.py)

Couche d'abstraction compatible `torch.utils.data.Dataset`, branchable directement sur un `DataLoader`.

**Ce qui est implémenté :**
- Instanciation avec `split_type` (train/val/test) et modèle cible
- Préparation automatique des splits au premier appel
- Construction automatique du vocabulaire si non fourni
- `__getitem__(idx)` → retourne `(image_tensor, caption_indices)`
- `get_all_captions_for_image(idx)` → retourne les 5 légendes de référence (pour l'évaluation BLEU)
- Chargement d'image en RGB via PIL + application des transforms

---

### Étape 6 — Point d'entrée de développement

**Fichier :** [`src/main.py`](src/main.py)

Script de test évolutif — mis à jour à chaque étape pour valider les nouveaux modules.

---

### Étape 7 — Encodeur CNN

**Fichier :** [`src/model/encoder.py`](src/model/encoder.py)

VGG19 pré-entraîné sur ImageNet, tronqué après le 4ème bloc convolutif (avant le dernier max-pooling), comme décrit en Sec. 3.1.1 & 4.3 du papier.

**Ce qui est implémenté :**
- Chargement de VGG19 avec poids `IMAGENET1K_V1`
- Conservation uniquement de `features[:29]` (blocs conv 1 à 4)
- `forward()` : `[B, 3, 224, 224]` → reshape → `[B, 196, 512]`
- `set_fine_tune()` : gèle ou libère les gradients CNN
- Poids gelés par défaut (fidèle au papier)

**Résultat validé :**
```
features : torch.Size([1, 196, 512])
  └── 196 = grille 14×14 sur l'image (chaque position = une région)
  └── 512 = description visuelle de cette région (annotation vector aᵢ)
```

---

### Étape 8 — Mécanisme d'attention (Soft)

**Fichier :** [`src/model/attention.py`](src/model/attention.py)

Attention déterministe (Sec. 4.2) — pour chaque pas de temps t, calcule un vecteur de contexte `ẑt` comme somme pondérée des annotation vectors.

**Ce qui est implémenté :**
```
eₜᵢ  = W2 · tanh(W1_a · aᵢ + W1_h · hₜ₋₁)   ← scoring MLP
αₜ   = softmax(eₜ)                              ← poids (somme = 1)
ẑₜ   = Σᵢ αₜᵢ · aᵢ                             ← contexte visuel
```
- Retourne `(z_hat, alpha)` — `alpha` sert à la doubly stochastic regularization pendant l'entraînement

**Résultat validé :**
```
z_hat    : torch.Size([1, 512])   ← contexte visuel pour le LSTM
alpha    : torch.Size([1, 196])   ← poids sur les 196 régions
alpha sum: 1.0000                 ← somme à 1 ✓
```

---

## 2. Schéma de l'état actuel vs prochaines étapes

```
╔══════════════════════════════════════════════════════════════════╗
║              PIPELINE COMPLET — SHOW, ATTEND AND TELL            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              ✅ RÉALISÉ — DATA PIPELINE                  │    ║
║  │                                                         │    ║
║  │  [Image brute]                                          │    ║
║  │       │                                                 │    ║
║  │       ▼                                                 │    ║
║  │  transforms_factory ──► Resize + Normalize (VGG19)      │    ║
║  │       │                                                 │    ║
║  │       ▼                                                 │    ║
║  │  AnnotationDataset ──► (image_tensor, caption_indices)  │    ║
║  │       │                        │                        │    ║
║  │       │                 vocabulary.encode()             │    ║
║  │       ▼                        │                        │    ║
║  │  DataLoader ◄───────────────────┘                       │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                         │                                        ║
║                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              ✅ RÉALISÉ — ENCODEUR CNN                   │    ║
║  │                                                         │    ║
║  │  VGG19 pré-entraîné (poids gelés)                       │    ║
║  │  [B, 3, 224, 224] ──► [B, 196, 512]                     │    ║
║  │  196 régions × 512 dim = annotation vectors aᵢ          │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                         │                                        ║
║             annotation vectors  aᵢ ∈ ℝ⁵¹²  (L=196 vecteurs)   ║
║                         │                                        ║
║                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │           ✅ RÉALISÉ — ATTENTION SOFT                    │    ║
║  │                                                         │    ║
║  │  eₜᵢ = W2·tanh(W1_a·aᵢ + W1_h·hₜ₋₁)                  │    ║
║  │  αₜ  = softmax(eₜ)          [B, 196] — somme = 1       │    ║
║  │  ẑₜ  = Σᵢ αₜᵢ · aᵢ         [B, 512] — contexte visuel │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                         │                                        ║
║                    context vector ẑₜ                             ║
║                         │                                        ║
║                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │           ✅ RÉALISÉ — DÉCODEUR LSTM                     │    ║
║  │                                                         │    ║
║  │  Initialisation :                                       │    ║
║  │    h₀ = MLP(mean(aᵢ))                                  │    ║
║  │    c₀ = MLP(mean(aᵢ))                                  │    ║
║  │                                                         │    ║
║  │  À chaque pas t :                                       │    ║
║  │    iₜ = σ(Wᵢ·Eyₜ₋₁ + Uᵢ·hₜ₋₁ + Zᵢ·ẑₜ)               │    ║
║  │    fₜ = σ(Wf·Eyₜ₋₁ + Uf·hₜ₋₁ + Zf·ẑₜ)                │    ║
║  │    cₜ = fₜ·cₜ₋₁ + iₜ·tanh(Wc·Eyₜ₋₁+Uc·hₜ₋₁+Zc·ẑₜ)  │    ║
║  │    oₜ = σ(Wo·Eyₜ₋₁ + Uo·hₜ₋₁ + Zo·ẑₜ)                │    ║
║  │    hₜ = oₜ · tanh(cₜ)                                  │    ║
║  │                                                         │    ║
║  │  Prédiction du mot :                                    │    ║
║  │    p(yₜ) ∝ exp(Lo·(E·yₜ₋₁ + Lh·hₜ + Lz·ẑₜ))          │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                         │                                        ║
║                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │         ❌ À IMPLÉMENTER — BOUCLE D'ENTRAÎNEMENT         │    ║
║  │                                                         │    ║
║  │  Loss : -log p(y|a) + λ·Σᵢ(1 - Σₜ αₜᵢ)²               │    ║
║  │  Optimiseur : Adam / RMSProp                            │    ║
║  │  Early stopping sur BLEU@val                            │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                         │                                        ║
║                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │         ❌ À IMPLÉMENTER — INFÉRENCE & ÉVALUATION        │    ║
║  │                                                         │    ║
║  │  Décodage : greedy search ou beam search                │    ║
║  │  Métriques : BLEU-1/2/3/4, METEOR                       │    ║
║  │  Visualisation : upsampling αₜᵢ × 16 + Gaussien        │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
```
Tout fonctionne parfaitement. Voici ce que confirme la sortie :


images   : (4, 3, 224, 224)   ✓ batch de 4 images normalisées
features : (4, 196, 512)       ✓ encoder → 196 régions × 512 dim
logits   : (4, 19, 2669)       ✓ 19 prédictions de mots × 2669 mots du vocab
alphas   : (4, 19, 196)        ✓ poids d'attention à chaque pas de temps
alpha sum: [1., 1., 1., 1.]    ✓ somme à 1 pour chaque image
vocab    : 2669 mots           ✓ vocabulaire construit depuis Flickr8k
Les 3 modules — Encoder → Attention → Decoder — sont connectés et produisent les bonnes dimensions. Le pipeline forward complet est opérationnel.

Maintenant on met à jour PROGRESSION.md et on passe à la boucle d'entraînement :

---

## 3. Prochaines étapes détaillées

### Étape A — Encodeur CNN (`src/model/encoder.py`)

Extraire les feature maps spatiales de VGG19 (couche `conv4_3`, avant le dernier max-pooling).

```
Input  : [B, 3, 224, 224]
Output : [B, 196, 512]   ← 14×14 positions aplaties, 512 canaux
```

- Charger VGG19 pré-entraîné (`torchvision.models.vgg19`)
- Supprimer les couches `classifier` (FC)
- Extraire après `features[28]` (4ème bloc conv avant pooling)
- Optionnel : couche de projection linéaire vers dimension `D` configurable

---

### Étape B — Mécanisme d'attention (`src/model/attention.py`)

Implémenter les deux variantes décrites dans le papier.

**Soft Attention (priorité 1) :**
```python
# MLP de scoring
e = W2 · tanh(W1_a · a + W1_h · h)   # [B, L]
alpha = softmax(e)                      # [B, L]
z_hat = (alpha.unsqueeze(2) * a).sum(1) # [B, D]
```

**Hard Attention (priorité 2) :**
- Échantillonner un emplacement depuis `Multinoulli({αᵢ})`
- Gradient via REINFORCE + baseline mobile

**Doubly Stochastic Regularization :**
```python
reg_loss = lambda * ((1 - alphas.sum(dim=1)) ** 2).sum()
```

---

### Étape C — Décodeur LSTM (`src/model/decoder.py`)

LSTM conditionné sur le contexte visuel à chaque pas de temps.

- Embedding de mots : `nn.Embedding(vocab_size, embed_dim)`
- LSTM cell avec 3 entrées : `E·yₜ₋₁`, `hₜ₋₁`, `ẑₜ`
- Initialisation `h₀, c₀` via deux MLPs sur `mean(aᵢ)`
- Deep output layer : `p(yₜ) ∝ exp(Lo·(E·yₜ₋₁ + Lh·hₜ + Lz·ẑₜ))`
- Gating scalar `βₜ = σ(f(hₜ₋₁))` pour pondérer le contexte visuel

---

### Étape D — Boucle d'entraînement (`src/train.py`)

```
Pour chaque batch (images, captions) :
  1. Encoder les images → features [B, 196, 512]
  2. Pour t = 1..T :
     a. Calculer αₜ et ẑₜ via attention
     b. Passer dans le LSTM → hₜ
     c. Prédire p(yₜ | ...)
  3. Calculer la loss cross-entropie + régularisation doubly stochastic
  4. Backprop + Adam step
  5. Loguer avec wandb
```

- Batching stratifié par longueur (comme dans le papier)
- Early stopping sur BLEU@validation
- Checkpointing du meilleur modèle

---

### Étape E — Évaluation (`src/evaluate.py`)

- Générer des légendes (greedy ou beam search) sur le split test
- Calculer BLEU-1/2/3/4 et METEOR contre les 5 références
- Visualiser les cartes d'attention (upsampling ×16 + filtre Gaussien)

---

## 4. Résumé de l'avancement

| Module | Statut | Fichier |
|---|---|---|
| Infrastructure projet | ✅ Complet | `pyproject.toml`, `Dockerfile` |
| Vocabulaire | ✅ Complet | `src/dataset/vocabulary.py` |
| Splits Flickr8k | ✅ Complet | `src/utils/data.py` |
| Transforms images | ✅ Complet | `src/dataset/transforms_factory.py` |
| Dataset PyTorch | ✅ Complet | `src/dataset/AnnotationDataset.py` |
| Splits MSCOCO | ⏳ Stub | `src/utils/data.py` |
| **Encodeur CNN** | ✅ Complet | `src/model/encoder.py` |
| **Attention (Soft)** | ✅ Complet | `src/model/attention.py` |
| **Attention (Hard)** | ❌ Manquant | `src/model/attention.py` |
| **Décodeur LSTM** | ✅ Complet | `src/model/decoder.py` |
| **Boucle d'entraînement** | ❌ Manquant | `src/train.py` |
| **Inférence / Beam search** | ❌ Manquant | `src/inference.py` |
| **Évaluation BLEU/METEOR** | ❌ Manquant | `src/evaluate.py` |
| **Visualisation attention** | ❌ Manquant | `src/visualize.py` |

**Avancement global : ~65% du papier implémenté**
