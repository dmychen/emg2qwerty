# CNN-BiLSTM CTC Architecture

## Motivation

The TDS-Conv baseline has parameter-efficient temporal convolutions that provide a fixed receptive field (~1s / 125 frames), but they cannot model dependencies beyond that window. Bidirectional LSTMs offer unbounded temporal context by maintaining hidden states across the full sequence, which is advantageous for capturing the overlapping co-articulation patterns inherent in continuous typing (preparatory muscle activation precedes keystrokes by tens of ms, and relaxation trails after). Prior work on sEMG-to-text tasks reports that LSTM encoders with CTC outperform TDS-Conv baselines by ~10% absolute CER in zero-shot settings.

## Architecture

The model reuses the existing preprocessing frontend (`SpectrogramNorm` + `MultiBandRotationInvariantMLP` + `Flatten`) and replaces `TDSConvEncoder` with a `CNNBiLSTMEncoder`.

### Pipeline

```
Input: (T, N, 2, 16, 33)  — spectrogram from STFT (125 Hz)
  |
  SpectrogramNorm          — per-channel BatchNorm2d across 32 channels
  |
  MultiBandRotationInvariantMLP — per-band MLP with rotation pooling
  |                                output: (T, N, 2, 384)
  Flatten(start_dim=2)    — (T, N, 768)
  |
  CNNBiLSTMEncoder:
    Conv1d(768→256, k=3, pad=1) + GELU + LayerNorm
    Conv1d(256→256, k=3, pad=1) + GELU + LayerNorm
    4-layer BiLSTM(input=256, hidden=256, dropout=0.3)
  |                                output: (T, N, 512)
  |
  Linear(512→97) + LogSoftmax  — 96 chars + CTC blank
```

### Design Decisions

- **Shallow CNN frontend**: Two Conv1d layers (kernel=3, stride=1, padding=1) reduce dimensionality from 768 to 256 and extract local temporal bigram/trigram features before the recurrent layers. Stride=1 with padding=1 preserves temporal length, so `emission_lengths = input_lengths` (no `T_diff` adjustment needed for CTC).
- **4-layer BiLSTM**: Captures forward and backward co-articulation dynamics. Hidden size of 256 per direction yields 512-dim output. 4 layers provide sufficient depth; empirical evidence shows diminishing returns beyond 4-6 layers for this task.
- **GELU activation**: Smoother gradient flow than ReLU in the CNN frontend.
- **LayerNorm** (not BatchNorm) in the CNN: More stable with variable-length sequences and small batch sizes at test time (test uses batch_size=1 on full sessions).
- **Inter-layer dropout (0.3)**: Applied between LSTM layers only (not to recurrent connections), which preserves long-term memory while regularizing the vertical depth.

### Parameter Count

| Component | Params |
|---|---|
| Frontend (SpectrogramNorm + MLP) | ~406K |
| Conv1d layers + norms | ~788K |
| 4-layer BiLSTM | ~5,775K |
| Output projection | ~50K |
| **Total** | **~7.0M** |

This is ~1.3x the TDS-Conv baseline (5.3M). The LSTM dominates at 82% of parameters.

## Training Strategy

### Optimizer & Schedule

- **Optimizer**: Adam (matching baseline). AdamW with weight_decay=1e-4 is a reasonable alternative if overfitting is observed.
- **Learning rate**: 3e-4 peak (reduced from baseline 1e-3; deep BiLSTMs are sensitive to aggressive LR and risk exploding gradients).
- **Schedule**: Linear warmup (10 epochs) + cosine annealing to 1e-6 (existing `linear_warmup_cosine_annealing` scheduler).
- **Gradient clipping**: `gradient_clip_val=5.0` via PyTorch Lightning trainer to prevent gradient explosion in BPTT.
- **Epochs**: 150 (baseline default).

### Regularization

- Inter-layer LSTM dropout: 0.3
- Data augmentation (existing pipeline): `RandomBandRotation`, `TemporalAlignmentJitter`, `SpecAugment`
- No additional regularization initially; weight decay can be introduced if val CER plateaus while train CER continues to drop.

### Hyperparameters Summary

| Parameter | Value |
|---|---|
| `cnn_hidden_dim` | 256 |
| `lstm_hidden_dim` | 256 |
| `lstm_layers` | 4 |
| `dropout` | 0.3 |
| `optimizer.lr` | 3e-4 |
| `batch_size` | 32 |
| `trainer.max_epochs` | 150 |
| `trainer.gradient_clip_val` | 5.0 |
| `window_length` | 8000 (4s) |
| `padding` | [1800, 200] |

## Files Modified

- `emg2qwerty/modules.py` — added `CNNBiLSTMEncoder` class
- `emg2qwerty/lightning.py` — added `BiLSTMCTCModule` class
- `config/model/bilstm_ctc.yaml` — Hydra config for the BiLSTM model

## Running

### CLI (local)

```bash
python -m emg2qwerty.train \
  model=bilstm_ctc \
  optimizer.lr=3e-4 \
  trainer.gradient_clip_val=5.0
```

### Colab

```python
# In a Colab cell, after cloning the repo and installing dependencies:
!pip install -e .

# Training
!python -m emg2qwerty.train \
  model=bilstm_ctc \
  optimizer.lr=3e-4 \
  trainer.gradient_clip_val=5.0 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  num_workers=2

# Evaluation only (with a checkpoint)
!python -m emg2qwerty.train \
  model=bilstm_ctc \
  train=False \
  checkpoint=path/to/best.ckpt \
  decoder=ctc_greedy
```

### Hydra Config Override (full example)

```bash
python -m emg2qwerty.train \
  model=bilstm_ctc \
  optimizer.lr=3e-4 \
  batch_size=32 \
  trainer.max_epochs=150 \
  trainer.gradient_clip_val=5.0 \
  lr_scheduler.scheduler.warmup_epochs=10 \
  decoder=ctc_greedy
```

To switch to beam search decoding with LM at test time:
```bash
python -m emg2qwerty.train \
  model=bilstm_ctc \
  train=False \
  checkpoint=path/to/best.ckpt \
  decoder=ctc_beam
```
