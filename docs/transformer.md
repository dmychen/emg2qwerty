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
    Conv1d (k=3, pad=1) + GELU  — smooths out high frequency noise
    Conv1d (k=3, pad=1) + GELU  — reduces dimensionality 
    PositionalEncoding          — adds sine/cosine timing signals
    4-layer TransformerEncoder  — nhead=16, dim_feedforward=2048, dropout=0.15
  |                                output: (T, N, 512)
  |
  Linear(512→97) + LogSoftmax  — 96 chars + CTC blank
```

## Design Decisions

- **The CNN Frontend:** The vanilla transformer failed because it got overwhelmed by thousands of noisy frames and had no sense of "local" muscle movement. Adding a 2-layer Conv1d with a kernel size of 3 and padding of 1 acts as a moving average that groups nearby time-steps into a cleaner feature representation. We kept padding=1 so the time dimension `T` doesn't shrink, making CTC alignment easier.
- **Positional Encoding:** Added standard sinusoidal encoding. Without this, the model has no concept of sequential order.
- **Layers & Heads (A100 Optimized):** Since we are no longer memory-constrained by the Colab T4, we increased the attention head count to 16, the model dimension to 512, and the feed-forward dimension to 2048. This allows the model to capture far more complex overlaps in co-articulation, making it highly competitive with the CNN baseline. We also bumped dropout to `0.15` to stabilize the deeper network.
- **60 Epoch Hyperparameters:** Instead of the default 150 epochs, we scaled down to 60. This required shrinking the warmup phase from 10 epochs to 4, and slightly bumping the peak learning rate to `5e-4` to force faster convergence.
- **Decoding Checkpoint:** We explicitly disabled the `decode_batch` call during the `train` step because CTC greedy/beam decoding on every batch takes an eternity. We only decode on `val` and `test`.

## Hyperparameters Summary (Scaled for 60 Epochs)

| Parameter | Value |
|---|---|
| `cnn_hidden_dim` | 512 |
| `nhead` | 16 |
| `num_layers` | 6 |
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
