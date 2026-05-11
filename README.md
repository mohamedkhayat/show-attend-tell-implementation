# Show, Attend and Tell

Implementation of *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"* (Xu et al., 2015) on Flickr8k, comparing three encoders — VGG19, ResNet50, and CLIP ViT-B/16 — all with the same soft-attention LSTM decoder.

## Results

Val-set BLEU after early stopping (patience=7 on BLEU-4):

| Encoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Best epoch |
|---------|--------|--------|--------|--------|-----------|
| Paper (VGG soft-attn) | 67.0 | 44.8 | 29.9 | 19.5 | — |
| VGG19 (ours) | 58.81 | 37.98 | 25.09 | 17.20 | 18 |
| ResNet50 | 63.41 | 42.56 | 28.90 | 19.67 | 9 |
| **CLIP ViT-B/16** | **69.39** | **50.17** | **36.09** | **25.89** | **8** |

CLIP ViT-B/16 is +33% on BLEU-4 vs the paper's VGG baseline and converges in 8 epochs. VGG19 is slightly below the paper because the paper fine-tunes the CNN encoder after epoch 20 (Section 5.3); our encoder stays frozen throughout.

---

## Setup

### 1. Start the container

```bash
docker compose up -d notebook
```

Jupyter is available at **http://localhost:8889** (no token). The container mounts this directory as `/workspace` and persists model weights in a named Docker volume.

### 2. Get Flickr8k

Download from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place files at:

```
data/flicker8k/
├── captions.txt
└── Images/
    ├── 1000268201_693b08cb0e.jpg
    └── ...
```

Train/val/test splits are generated automatically on first run.

### 3. Install dependencies (first run only)

The container installs from `requirements.txt` on startup. If running a pre-existing container, install manually:

```bash
docker compose exec notebook pip install -r requirements.txt
```

---

## CLI Reference

All commands run inside the container:

```bash
docker compose exec notebook bash
# then:
python -m src.main <overrides>
```

Or as a one-liner:

```bash
docker compose exec notebook python -m src.main <overrides>
```

### Training

```bash
# VGG19 (paper baseline)
python -m src.main mode=train encoder=vgg19

# ResNet50
python -m src.main mode=train encoder=resnet50

# CLIP ViT-B/16
python -m src.main mode=train encoder=clip_vit_b16
```

Checkpoints are saved to `checkpoints/<encoder>/best.pt` whenever val BLEU-4 improves.

### Evaluation (test-set BLEU)

```bash
python -m src.main mode=eval encoder=vgg19
python -m src.main mode=eval encoder=resnet50
python -m src.main mode=eval encoder=clip_vit_b16
```

### Caption a single image

```bash
python -m src.main mode=caption encoder=vgg19 \
    caption.image_path=data/flicker8k/Images/some_image.jpg
```

### Common overrides

Any config key can be overridden on the command line (Hydra syntax):

```bash
# Smaller batch for debugging
python -m src.main encoder=resnet50 data.batch_size=32

# Custom learning rate
python -m src.main encoder=vgg19 training.lr=1e-4

# Log to W&B online (requires wandb login first)
python -m src.main encoder=clip_vit_b16 wandb.mode=online

# Quick smoke-test (1 epoch)
python -m src.main encoder=vgg19 training.epochs=1 data.batch_size=16

# Disable mixed precision
python -m src.main training.mixed_precision=false

# Enable torch.compile (faster, but slower first epoch)
python -m src.main training.compile=true
```

---

## Configuration

The base config lives at `config/train.yaml`. Encoder-specific settings are in `config/encoder/`.

| Key | Default | Description |
|-----|---------|-------------|
| `encoder` | `vgg19` | Encoder backbone: `vgg19`, `resnet50`, `clip_vit_b16` |
| `data.batch_size` | `128` | Batch size (safe for RTX 5090 32 GB across all encoders) |
| `data.max_seq_len` | `30` | Maximum caption length (tokens) |
| `data.num_workers` | `0` | DataLoader workers (0 avoids Docker `/dev/shm` errors) |
| `training.epochs` | `50` | Max epochs (early stopping usually triggers earlier) |
| `training.lr` | `4e-4` | RMSProp learning rate |
| `training.patience` | `7` | Early stopping patience on val BLEU-4 |
| `training.lambda_reg` | `1.0` | Doubly stochastic regularisation weight |
| `training.mixed_precision` | `true` | bf16 AMP (Blackwell / RTX 5090 native) |
| `training.compile` | `false` | `torch.compile` on decoder |
| `wandb.mode` | `offline` | `offline` / `online` / `disabled` |

---

## Notebook

`notebooks/train_eval.ipynb` (at http://localhost:8889/notebooks/train_eval.ipynb) contains:

1. Encoder shape smoke-tests (all 3 → `(B, 196, D)`)
2. Full training cells for each encoder (calls the CLI via subprocess)
3. BLEU comparison table across all 3 models
4. Attention visualisation — per-word 14×14 soft-attention maps overlaid on images

---

## Architecture

All encoders output `(B, 196, D)` — a 14×14 spatial grid matching the paper.

| Encoder | Layer tapped | D |
|---------|-------------|---|
| VGG19 | `features[:29]` (4th conv block, before final pool) | 512 |
| ResNet50 | After `layer3` (before layer4 + avgpool) | 1024 |
| CLIP ViT-B/16 | Patch tokens from `CLIPVisionModel`, CLS dropped | 768 |

The decoder is a single LSTMCell with:
- Soft attention with β-gating: `z_t = σ(f_β(h_{t-1})) · Σ α_ti · a_i`
- Deep output layer: `p(y_t) ∝ exp(L_o(E·y_{t-1} + L_h·h_t + L_z·z_t))`
- Doubly stochastic regularisation: `λ · Σ_L(1 − Σ_t α_ti)²`
- Teacher forcing during training, greedy decoding at inference
