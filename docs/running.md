# Run STM

Install `.[encoders]`, supply the upstream repositories and encoder checkpoints
listed in [bank building](banks.md), then run `python -m silence_the_mimic --help`.
The installed `stm` command is equivalent. No assets are downloaded automatically.

```bash
python -m silence_the_mimic --model freevc --model-root /path/to/FreeVC \
  --checkpoint /path/to/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt \
  --bank artifacts/freevc_bank.pt --input /path/to/clean_wavs \
  --output artifacts/protected --device cuda:0 --iterations 80 --seed 42
```

For the other models, change `--model`, `--model-root`, `--checkpoint` and `--bank`
to the matching values from the bank commands. TriAAN-VC additionally requires
`--cpc-checkpoint`. GSV uses the original 704-bin checkpoint, not a 1025-bin model.
Run different upstream models in separate processes to avoid module-name collisions.

Input must already be mono, 16 kHz and within [-1, 1]. No implicit input
resampling, downmixing, normalization or clipping is performed. A directory
processes `.wav` files recursively in sorted relative-path order and preserves
the subdirectories; its output must be outside the input directory. Each file
uses `--seed + file_index`. Errors stop the batch, retaining completed outputs
rather than silently skipping a failed input. Existing WAVs or receipts are never
overwritten, so use a fresh output directory for a new run.

The default has no trim. `--trim-top-db 20` explicitly enables librosa trimming
and records the retained indices. This is an optional condition, not the default.
FLOAT WAV output preserves the core's float32 waveform without clipping or
normalization, including possible overshoot. `--output-subtype PCM_16` explicitly
requests quantization and rejects overshoot rather than silently clipping it.
FLOAT is different from the older research runner's PCM16 export.

| Model | Optimization mode | STM input / internal encoder rate |
| --- | --- | --- |
| FreeVC | train, weights frozen | 16 / 16 kHz |
| QuickVC | train, weights frozen | 16 / 16 kHz |
| TriAAN-VC | speaker encoder and CPC train, weights frozen | 16 / 16 kHz |
| GPT-SoVITS | eval, weights frozen | 16 / 32 kHz, differentiable resampling |

These modes follow the original STM optimization runner, not the bank-side eval
default. Train mode changes layer behavior; it does not optimize model weights.
BatchNorm running buffers may change in memory; no checkpoint is saved.

Each WAV has a `.wav.json` receipt recording the encoder/checkpoint/source and
bank hashes, STM config, target speaker, seed, preprocessing, output format and
versions. If a model-specific bank receipt exists, its model, encoder source,
checkpoint and bank input rate are checked. Older tensor-only banks load with an
explicit warning: matching dimensions alone cannot verify encoder compatibility.
Recorded wall time covers the synchronized `STM.protect` call, excluding model
loading and audio I/O; without warmup this is not a formal latency benchmark.
The runner does not override PyTorch's deterministic/backend settings. GPU
backward operations can be nondeterministic even with the same seed; backend
settings are recorded, and bit-identical GPU output is not guaranteed.

## Python API

For a custom differentiable encoder, the core-only installation is sufficient:
`python -m pip install .`. Supply `encode`, `wav` and `bank` using your encoder's
implementation:

```python
from silence_the_mimic import STM, STMConfig

# encode: waveform -> embedding, preserving gradients to the waveform.
# wav: mono float32 NumPy array or Tensor, within [-1, 1], at 16 kHz.
# bank: {speaker_id: embedding}, using the same encoder and preprocessing.
stm = STM(encode, STMConfig(iterations=80), device="cpu")
result = stm.protect(wav, sample_rate=16000, referral_bank=bank, seed=42)
protected_wav = result.waveform.cpu().numpy()
```

Use `device="cuda"` for GPU execution, with the encoder on the same device.
The default bank selection requires at least six embeddings and chooses the
sixth-farthest by MSE. For a fixed target, replace `referral_bank=bank` with
`target_embedding=target`; do not supply both.

The STM core does not automatically resample, trim, normalize or clip input
audio. Encoder adapters may perform their own internal preprocessing; for
example, the GSV attack adapter differentiably resamples 16-kHz input to 32 kHz.
Configuration options are defined in [STMConfig](../silence_the_mimic/core.py).
Set `tv_weight=0` to disable temporal regularization.

With the `.[encoders]` dependencies, a built-in adapter can replace your own
`encode`:

```python
from silence_the_mimic import create_attack_encoder, STM

encoder = create_attack_encoder("freevc", model_root="/path/to/FreeVC",
                                checkpoint="/path/to/encoder.pt", device="cuda:0")
stm = STM(encoder, device="cuda:0")
# Supply preprocessed wav and a matching bank as above.
result = stm.protect(wav, sample_rate=16000, referral_bank=bank, seed=42)
```

## Model-free example

For a runnable CPU example without model downloads:

```bash
PYTHONPATH=. python examples/minimal.py
```

This toy example checks functionality, not voice-cloning protection performance.
