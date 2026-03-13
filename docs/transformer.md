# Transformer & ConvTransformer CTC Architecture

## Overview
This documents our attempt to adapt Transformer encoders for raw sEMG signals. Transformers are great for long-range dependencies, but applying them directly to 125Hz noisy biological signals proved challenging. 

We built two versions:
1. **Vanilla Transformer (Failed Experiment):** A standard Positional Encoded Transformer that struggled to learn alignment due to the lack of local inductive bias.
2. **ConvTransformer (Successful Fix):** A hybrid model that adds a 1D Convolutional frontend to extract local muscle twitches *before* feeding them into the Transformer for global context modeling.

## Architecture

Both models reuse the `SpectrogramNorm` and `MultiBandRotationInvariantMLP` for initial feature extraction.

### ConvTransformer Pipeline

```
Input: (T, N, 2, 16, 33)  — spectrogram from STFT
  |
  SpectrogramNorm          — per-channel BatchNorm2d
  |
  MultiBandRotationInvariantMLP — per-band MLP with rotation pooling
  |                                output: (T, N, 2, 384)
  Flatten(start_dim=2)    — (T, N, 768)
  | 
  ConvTransformerEncoderModel:
    Conv1d (k=1)  — projects features down to 256
    2x ResNet1dBlocks (k=3, pad=1) + GELU + LayerNorm  — extracts local twitches
    PositionalEncoding          — adds sine/cosine timing signals
    4-layer TransformerEncoder  — nhead=8, dim_feedforward=512, dropout=0.1
  |                                output: (T, N, 256)
  |
  Linear(256→97) + LogSoftmax  — 96 chars + CTC blank
```

## Design Decisions

- **ResNet CNN Frontend:** To avoid 60-epoch starvation, we forced the model to learn 90% of the muscle patterns instantly by adding a deep, 4-layer ResNet-style 1D CNN block. It uses residual skip connections and LayerNorms, forcing rapid convergence on local features. Fast local learning means the Transformer backend can do less heavy-lifting early on.
- **Positional Encoding:** Added standard sinusoidal encoding. Without this, the model has no concept of sequential order.
- **Thinner Transformer & norm_first:** To combat exploding/vanishing gradients and data starvation in only 60 epochs, we reverted the baseline back to a thinner transformer (4 layers, 8 heads, 256 dim). Additionally, applying `norm_first=True` allows standard attention to converge far faster from scratch since gradients flow cleanly.
- **60 Epoch Hyperparameters:** Instead of the default 150 epochs, we scaled down to 60. This requires shrinking the warmup phase from 10 epochs to 4, and slightly bumping the peak learning rate to `5e-4` to force faster convergence.
- **Decoding Checkpoint:** We explicitly disabled the `decode_batch` call during the `train` step because CTC greedy/beam decoding on every batch takes an eternity. We only decode on `val` and `test`.

## Hyperparameters Summary (Scaled for 60 Epochs)

| Parameter | Value |
|---|---|
| `cnn_hidden_dim` | 256 |
| `nhead` | 8 |
| `num_layers` | 4 |
| `trainer.max_epochs` | 60 |
| `optimizer.lr` | 5e-4 |
| `warmup_epochs` | 4 |
| `batch_size` | 32 |

## Important Files

- `emg2qwerty/modules.py` — added `PositionalEncoding`, `TransformerEncoderModel` (vanilla), and `ConvTransformerEncoderModel`
- `emg2qwerty/lightning.py` — added `TransformerCTCModule` and `ConvTransformerCTCModule`
- `config/model/conv_transformer_ctc.yaml` — Hydra config

## Running in Colab

To run the successful ConvTransformer model, use these overrides to scale for a 60-epoch run:

```bash
!python -m emg2qwerty.train \
  user="single_user" \
  model=conv_transformer_ctc \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=60 \
  batch_size=32 \
  optimizer.lr=5e-4 \
  lr_scheduler.scheduler.warmup_epochs=4 \
  +trainer.gradient_clip_val=5.0
```
