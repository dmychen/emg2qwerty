# emg2qwerty Codebase Documentation

## 1. Concepts

### 1.1 The Learning Task

The model performs **sequence-to-sequence transcription from EMG signals to typed characters**. Two EMG wristbands (left and right, 16 electrodes each) record muscle activity at 2kHz while a user types on a keyboard. The model must learn the mapping from continuous EMG signal windows to discrete character sequences — the text the user typed during that window.

This is framed as a **temporal classification** problem: a variable-length input signal produces a variable-length character output, without explicit alignment between signal frames and characters. The character set has 96 classes (a-z, A-Z, 0-9, punctuation, space, enter, backspace, shift) plus a CTC blank token (97 total). Evaluation uses Character Error Rate (CER) computed via Levenshtein edit distance.

### 1.2 CNN Model Architecture (TDS-Conv)

The baseline model is a **Time-Depth Separable (TDS) Convolutional** encoder, from the paper *"Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions"* (Hannun et al., 2019). The full pipeline:

1. **SpectrogramNorm**: Per-channel 2D batch normalization across the 32 electrode channels (2 bands x 16 electrodes). Normalizes over (N, freq, time) slices.

2. **MultiBandRotationInvariantMLP**: A separate MLP per band (left/right). Each MLP applies the same linear layers across multiple rotational shifts of the 16 electrodes (offsets -1, 0, +1), then mean-pools over shifts. This provides invariance to electrode ring rotation on the wrist. Input per band: `(T, N, 16, 33)` → output: `(T, N, 384)`.

3. **Flatten**: Concatenates left and right band outputs → `(T, N, 768)`.

4. **TDSConvEncoder**: 4 blocks, each containing:
   - `TDSConv2dBlock`: A 2D temporal convolution (kernel width 32) with ReLU, residual connection, and layer norm. The "2D" treats channels and width as spatial dimensions while convolving only over time.
   - `TDSFullyConnectedBlock`: A 2-layer FC network with ReLU, residual connection, and layer norm.

5. **Linear → LogSoftmax**: Projects to 97 classes and produces log-probabilities for CTC.

The 4 TDS blocks each have kernel width 32, giving a total temporal receptive field of ~125 frames. The convolutions are causal-like (no padding), so the output sequence is shorter than the input by this receptive field.

### 1.3 CTC Loss

**Connectionist Temporal Classification (CTC)** solves the alignment problem between input frames and output characters. Key properties:

- Introduces a **blank token** (class 96) that the model outputs when no character is being typed. The model outputs a probability distribution over 97 classes at every time frame.
- A **many-to-one mapping** collapses model output to text: remove blanks, then collapse consecutive duplicate characters. E.g., `--hh-e-ll--l-oo--` → `hello` (where `-` is blank).
- CTC loss marginalizes over **all valid alignments** between the output sequence and target text, computing the total probability of producing the target via any alignment path.
- Requires that the output sequence length ≥ target text length (with blanks between repeated characters). The `emission_lengths` passed to CTCLoss are adjusted for the receptive field reduction.

---

## 2. Codebase Structure

### 2.1 Preprocessing

| What | Where |
|------|-------|
| Raw EMG data loading (HDF5 files) | `emg2qwerty/data.py` — `EMGSessionData` |
| Windowing, padding, jittering | `emg2qwerty/data.py` — `WindowedEMGDataset` |
| Batch collation (pad + length tracking) | `emg2qwerty/data.py` — `WindowedEMGDataset.collate()` |
| Transform pipeline (ToTensor, STFT, augmentations) | `emg2qwerty/transforms.py` |
| Transform config (which transforms, params) | `config/transforms/log_spectrogram.yaml` |
| Label extraction and normalization | `emg2qwerty/data.py` — `LabelData` |
| Character set definition and encoding | `emg2qwerty/charset.py` — `CharacterSet` |

### 2.2 Model & Loss

| What | Where |
|------|-------|
| Full model assembly + CTC loss + training loop | `emg2qwerty/lightning.py` — `TDSConvCTCModule` (line 140) |
| SpectrogramNorm | `emg2qwerty/modules.py` — `SpectrogramNorm` (line 13) |
| Rotation-invariant MLP (per-band) | `emg2qwerty/modules.py` — `RotationInvariantMLP` (line 46), `MultiBandRotationInvariantMLP` (line 113) |
| TDS convolutional blocks | `emg2qwerty/modules.py` — `TDSConv2dBlock` (line 172), `TDSFullyConnectedBlock` (line 216) |
| TDS encoder (stacks all blocks) | `emg2qwerty/modules.py` — `TDSConvEncoder` (line 243) |
| Model config (features, channels, kernel) | `config/model/tds_conv_ctc.yaml` |
| CTC loss instantiation | `emg2qwerty/lightning.py` line 183 |
| Forward pass + loss computation | `emg2qwerty/lightning.py` — `_step()` (line 200) |

### 2.3 Testing & Evaluation

| What | Where |
|------|-------|
| CER/IER/DER/SER metrics | `emg2qwerty/metrics.py` — `CharacterErrorRates` |
| CTC greedy decoder | `emg2qwerty/decoder.py` — `CTCGreedyDecoder` (line 107) |
| CTC beam decoder with LM | `emg2qwerty/decoder.py` — `CTCBeamDecoder` (line 370) |
| Metric update during training/val/test | `emg2qwerty/lightning.py` — `_step()` lines 232-238 |
| Test dataloader (full sessions, batch_size=1) | `emg2qwerty/lightning.py` — `test_dataloader()` (line 124) |
| Unit tests | `emg2qwerty/tests/` — `charset_test.py`, `data_test.py`, `decoder_test.py` |
| Decoder configs | `config/decoder/ctc_greedy.yaml`, `config/decoder/ctc_beam.yaml` |

### 2.4 Other

| What | Where |
|------|-------|
| Training entry point (Hydra main) | `emg2qwerty/train.py` — `main()` |
| DataModule (dataloaders, splits) | `emg2qwerty/lightning.py` — `WindowedEMGDataModule` (line 32) |
| Optimizer/scheduler instantiation | `emg2qwerty/utils.py` |
| Base config (all defaults) | `config/base.yaml` |
| Optimizer config | `config/optimizer/adam.yaml` |
| LR scheduler configs | `config/lr_scheduler/` |
| Dataset split definitions | `config/user/single_user.yaml`, `config/user/generic.yaml` |
| Cluster/compute configs | `config/cluster/local.yaml`, `config/cluster/slurm.yaml` |
| Dataset generation scripts | `scripts/generate_splits.py`, `scripts/print_dataset_stats.py` |

---

## 3. Technical Overview

### 3.1 Configuration System (Hydra)

The project uses **Hydra** with **OmegaConf** for hierarchical configuration. The entry point `train.py` is decorated with `@hydra.main(config_path="../config", config_name="base")`.

**Config composition**: `config/base.yaml` declares defaults that pull in sub-configs:
```yaml
defaults:
  - user: single_user           # dataset splits
  - transforms: log_spectrogram # preprocessing pipeline
  - model: tds_conv_ctc         # model + datamodule targets
  - optimizer: adam
  - lr_scheduler: linear_warmup_cosine_annealing
  - decoder: ctc_greedy
  - cluster: local
```

Each sub-config uses `# @package _global_` to merge into the top-level namespace. Hydra's `instantiate()` reads `_target_` fields to construct Python objects directly from config.

**To add a new model**: Create a new YAML in `config/model/` specifying `module._target_` (the LightningModule class) and `datamodule._target_` (the DataModule class), along with their constructor arguments. Override at the command line with `model=your_new_config`.

**To change hyperparameters at the command line**:
```bash
python -m emg2qwerty.train batch_size=64 optimizer.lr=5e-4 trainer.max_epochs=200
```

**Key defaults**: batch_size=32, lr=1e-3, max_epochs=150, seed=1501, warmup_epochs=10, monitor_metric=val/CER.

### 3.2 Data Preprocessing

**Raw signal**: Two EMG bands (left/right wrist), each 16 electrode channels at 2kHz, stored in HDF5 files. Each session is a few minutes of typing.

**Windowing** (`WindowedEMGDataset`): Sessions are split into overlapping windows of `window_length=8000` samples (4 seconds). Each window is expanded by `padding=(1800, 200)` — 900ms of past context and 100ms of future context — to give the convolutions sufficient temporal context at window edges. Labels are extracted only for the un-padded region using keystroke timestamps. During training, `jitter=True` randomly shifts window offsets for variation.

**Transform pipeline** (train):
1. `ToTensor`: Extracts `emg_left` and `emg_right` from the structured numpy array, stacks them → shape `(T, 2, 16)`.
2. `RandomBandRotation`: Randomly shifts electrode channels by -1, 0, or +1 positions per band (simulates ring rotation on the wrist).
3. `TemporalAlignmentJitter`: Random left-right temporal offset up to 60ms (120 samples at 2kHz), simulating imperfect synchronization.
4. `LogSpectrogram`: STFT with `n_fft=64`, `hop_length=16`. Produces log10-magnitude spectrogram. This **downsamples time by 16x** (2kHz → 125Hz) and produces 33 frequency bins (`n_fft//2 + 1`). Output: `(T', 2, 16, 33)` where `T' ≈ T/16`.
5. `SpecAugment`: Random time and frequency masking (up to 3 time masks of 25 frames, 2 freq masks of 4 bins).

Val/test pipelines use only `ToTensor` → `LogSpectrogram` (no augmentation).

**Collation**: `pad_sequence` pads variable-length inputs/targets along time; `input_lengths` and `target_lengths` tensors are returned for CTC loss computation.

### 3.3 Testing & Evaluation

**Metric**: `CharacterErrorRates` (`metrics.py`) computes CER using Levenshtein edit operations between predicted and target text, broken down into insertion/deletion/substitution rates. It is a `torchmetrics.Metric` subclass with distributed reduction support. Updated per-batch, computed per-epoch.

**Decoding**: After the forward pass, emissions (log-probabilities) are decoded into text. Two decoders are available:
- **CTCGreedyDecoder**: Argmax at each timestep, collapse blanks and consecutive duplicates. Fast, no external dependencies.
- **CTCBeamDecoder**: Beam search (beam_size=50) with a KenLM 6-gram character-level language model. Handles backspace/delete keys by maintaining separate label and LM tries. Significantly improves CER at inference cost.

**Test-time behavior**: Test dataset feeds entire sessions at once (no windowing, batch_size=1), providing a more realistic evaluation than windowed validation.

**Evaluation flow in `_step()`** (`lightning.py:200`):
1. Forward pass → emissions `(T, N, 97)`.
2. Compute `emission_lengths = input_lengths - T_diff` where `T_diff` accounts for temporal reduction from the conv encoder's receptive field.
3. CTCLoss on emissions vs. targets.
4. Decode emissions → predicted `LabelData`.
5. Update `CharacterErrorRates` metric with (prediction, target) pairs.
6. At epoch end, `_epoch_end()` calls `metrics.compute()` and logs CER/IER/DER/SER.

**Running evaluation only** (no training):
```bash
python -m emg2qwerty.train train=False checkpoint=path/to/model.ckpt
```

### 3.4 Model Definition & Where to Make Changes

The model is assembled in `TDSConvCTCModule.__init__()` (`lightning.py:144`) as an `nn.Sequential`:

```python
self.model = nn.Sequential(
    SpectrogramNorm(channels=32),
    MultiBandRotationInvariantMLP(in_features=528, mlp_features=[384]),
    nn.Flatten(start_dim=2),
    TDSConvEncoder(num_features=768, block_channels=[24,24,24,24], kernel_width=32),
    nn.Linear(768, 97),
    nn.LogSoftmax(dim=-1),
)
```

**To modify or replace the model**:

1. **New module classes**: Add them to `modules.py`. All modules receive input in **time-first format** `(T, N, features)`. Ensure your module's output maintains this convention.

2. **New LightningModule**: Create a new class in `lightning.py` (or a new file). It must:
   - Accept `optimizer`, `lr_scheduler`, `decoder` DictConfig args (passed from Hydra).
   - Implement `forward()` returning `(T, N, num_classes)` log-probabilities.
   - Implement the `_step()` logic or reuse the existing one. The key constraint is that emissions must be `(T, N, num_classes)` shaped for CTCLoss and the decoders.
   - Call `self.ctc_loss(log_probs, targets, input_lengths, target_lengths)`.
   - Handle `emission_lengths` — if your model changes the temporal dimension (e.g., via striding or pooling), you must compute output lengths accordingly. The current code assumes no temporal downsampling beyond the conv receptive field: `emission_lengths = input_lengths - T_diff`.

3. **New model config**: Create `config/model/your_model.yaml`:
   ```yaml
   # @package _global_
   module:
     _target_: emg2qwerty.lightning.YourModule
     # your constructor args here
   datamodule:
     _target_: emg2qwerty.lightning.WindowedEMGDataModule
     window_length: 8000
     padding: [1800, 200]
   ```
   Run with `python -m emg2qwerty.train model=your_model`.

4. **For RNN-based models**: The input after preprocessing is `(T, N, 2, 16, 33)`. You can keep `SpectrogramNorm` and `MultiBandRotationInvariantMLP` as the front-end (they produce `(T, N, 768)` after flattening), then replace `TDSConvEncoder` with your RNN. Alternatively, reshape/project the raw spectrogram features directly into an RNN. Key consideration: CTC requires that output length ≥ target length, so avoid aggressive temporal downsampling.

5. **Hyperparameter changes** (no code changes needed): Override via command line or edit config YAMLs. Examples:
   - Learning rate: `optimizer.lr=5e-4`
   - Batch size: `batch_size=64`
   - Scheduler: `lr_scheduler=cosine_annealing`
   - Decoder: `decoder=ctc_beam`
   - TDS channels: `module.block_channels=[32,32,32,32]`
   - MLP features: `module.mlp_features=[512]`
