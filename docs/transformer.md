# Transformer CTC Architecture

## Overview
This is a baseline implementation of a Transformer encoder for raw sEMG signals. Transformers are popular for their ability to model long-range dependencies across a whole sequence at once without the bottleneck of recurrence (like LSTMs) or limited receptive fields (like CNNs).

We use a standard PyTorch `TransformerEncoder` hooked up to the CTC loss. Since transformers do not inherently process data in order, we also add a basic sinusoidal Positional Encoding layer before passing the features into the attention blocks.

## Architecture

The model reuses the existing `SpectrogramNorm` and `MultiBandRotationInvariantMLP` for feature extraction, flatten the channels, and then passes the sequence into the `TransformerEncoderModel`.

### Pipeline

```
Input: (T, N, 2, 16, 33)  — spectrogram from STFT
  |
  SpectrogramNorm          — per-channel BatchNorm2d
  |
  MultiBandRotationInvariantMLP — per-band MLP with rotation pooling
  |                                output: (T, N, 2, 384)
  Flatten(start_dim=2)    — (T, N, 768)
  | 
  TransformerEncoderModel:
    PositionalEncoding (adds sine/cosine timing signals)
    4-layer TransformerEncoder (nhead=8, dim_feedforward=512, dropout=0.1)
  |                                output: (T, N, 768)
  |
  Linear(768→97) + LogSoftmax  — 96 chars + CTC blank
```

## Design Decisions

- **No Downsampling:** Unlike convolutions, standard self-attention doesn't compress the time dimension. The output sequence length matches the input length exactly (`emission_lengths = input_lengths`).
- **Positional Encoding:** Standard sinusoidal encoding was implemented. Without this, the model would treat the sequence as an unordered "bag of frames" and fail completely on a time-series task like typing.
- **Layers & Heads:** Defaults to a relatively small transformer (4 layers, 8 heads, 512 feed-forward dim) to prevent out-of-memory errors on Google Colab's T4 GPUs, since attention scales quadratically with sequence length ($O(T^2)$). Keep in mind our window lengths are 8000 frames (4 seconds).
- **Decoding Checkpoint:** In the PyTorch Lightning module `_step` function, we explicitly disabled the `decode_batch` call during training (`if phase in ["val", "test"]`). Performing CTC decoding on every single training batch is extremely slow and unnecessary for calculating loss.

## Hyperparameters Summary

| Parameter | Value |
|---|---|
| `nhead` | 8 |
| `num_layers` | 4 |
| `dim_feedforward` | 512 |
| `dropout` | 0.1 |
| `window_length` | 8000 (4s) |
| `padding` | [0, 0] (no explicit context padding needed) |

## Important Files

- `emg2qwerty/modules.py` — added `PositionalEncoding` and `TransformerEncoderModel`
- `emg2qwerty/lightning.py` — added `TransformerCTCModule`
- `config/model/transformer_ctc.yaml` — Hydra config

## Running in Colab

Switch the `model` arg to `transformer_ctc` and enable GPU. Since memory might be tight, reducing `batch_size` is recommended if you hit OOM errors.

```bash
!python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_ctc \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=40 \
  batch_size=16
```
