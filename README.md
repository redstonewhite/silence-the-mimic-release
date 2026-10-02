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

FreeVC example: prepare its upstream repository, speaker-encoder checkpoint
and VCTK `wav48`. Run the commands below from this repository's root.

First build a referral bank (speaker embeddings used to select the target):

```bash
PYTHONPATH=. python examples/build_bank.py --model freevc \
  --model-root /path/to/FreeVC \
  --checkpoint /path/to/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt \
  --vctk-root /path/to/VCTK-Corpus/wav48 --output artifacts/freevc_bank.pt
```

The builder selects `_023.wav` per speaker (`_021.wav` for p268/p295/p340)
and requires at least six speakers.

Then protect an audio file using the same encoder and bank:

```bash
python -m silence_the_mimic --model freevc --model-root /path/to/FreeVC \
  --checkpoint /path/to/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt \
  --bank artifacts/freevc_bank.pt --input clean.wav --output protected.wav \
  --device cuda:0
```

Input must already be mono 16 kHz, within [-1, 1]. Default: 80 iterations, no trimming,
float32 WAV output, and no overwriting. Directories are also supported.

For `quickvc`, `triaanvc` or `gpt_sovits`, use matching model paths, weights and
banks. QuickVC/TriAAN-VC need extracted speaker-encoder weights, not full VC
checkpoints; TriAAN-VC also needs `--cpc-checkpoint`. GPT-SoVITS needs the
704-input-bin, 512-output-dimension MelStyleEncoder, not the 1025-bin variant.

More options: `python -m silence_the_mimic --help` and
`python examples/build_bank.py --help`.
For a custom encoder, see [the minimal example](examples/minimal.py).

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
