# Silence-the-Mimic

Core implementation of STM: optimize log-STFT magnitudes with psychoacoustic
bounds and per-step projection, while keeping the original phase fixed.

[Paper](https://arxiv.org/abs/2610.00662)

Includes encoder adapters for FreeVC, QuickVC, TriAAN-VC and GPT-SoVITS.
Model checkpoints, cloning systems and evaluation pipelines are not included.

## Installation

Requires Python 3.10+, NumPy and PyTorch.

```bash
python -m pip install ".[encoders]"
```

## Usage

First prepare the upstream encoder and [build a matching referral bank](docs/banks.md).
Then protect a mono 16-kHz audio file:

```bash
python -m silence_the_mimic --model freevc --model-root /path/to/FreeVC \
  --checkpoint /path/to/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt \
  --bank artifacts/freevc_bank.pt --input clean.wav --output protected.wav \
  --device cuda:0
```

Input must already be mono 16 kHz. Default: 80 iterations, no trimming,
float32 WAV output, and no overwriting. Directories are also supported.
See [runner settings](docs/running.md) for other models and options, or the
[Python API](docs/running.md#python-api) for a custom encoder.

## License

[MIT](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
psychoacoustic code attribution and permission.

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{xu2026silencethemimic,
  title         = {Silence-the-Mimic: Accelerating Imperceptible Perturbation Generation Against Voice Cloning},
  author        = {Runqiu Xu},
  year          = {2026},
  eprint        = {2610.00662},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2610.00662}
}
```
