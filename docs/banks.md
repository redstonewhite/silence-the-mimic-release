# Original-model referral banks

Install the optional adapter dependencies:

```bash
python -m pip install ".[encoders]"
```

Supply VCTK `wav48`, the upstream model checkouts, and the original speaker
checkpoints. No audio, model repository or checkpoint is downloaded by these
scripts. Paths below are placeholders, not bundled assets.

```bash
PYTHONPATH=. python examples/build_bank.py --model freevc \
  --model-root /path/to/FreeVC \
  --checkpoint /path/to/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt \
  --vctk-root /path/to/VCTK-Corpus/wav48 --output artifacts/freevc_bank.pt

PYTHONPATH=. python examples/build_bank.py --model quickvc \
  --model-root /path/to/QuickVC-VoiceConversion \
  --checkpoint /path/to/quickvc_spk_enc.pth \
  --vctk-root /path/to/VCTK-Corpus/wav48 --output artifacts/quickvc_bank.pt

PYTHONPATH=. python examples/build_bank.py --model triaanvc \
  --model-root /path/to/TriAAN-VC --checkpoint /path/to/triaanvc_spk_enc.pth \
  --cpc-checkpoint /path/to/TriAAN-VC/cpc/cpc.pt \
  --vctk-root /path/to/VCTK-Corpus/wav48 --output artifacts/triaanvc_bank.pt

PYTHONPATH=. python examples/build_bank.py --model gpt_sovits \
  --model-root /path/to/GPT-SoVITS --checkpoint /path/to/MelStyleEncoder.pth \
  --vctk-root /path/to/VCTK-Corpus/wav48 --output artifacts/gsv_bank.pt
```

Use `--device cuda:0` for GPU execution. QuickVC and TriAAN-VC require extracted
speaker-encoder state dictionaries, not a full VC model. GSV requires the original
704-input-bin, 512-output-dimension `MelStyleEncoder` checkpoint; an official
1025-input-bin checkpoint is not interchangeable and is explicitly rejected.

The original selection is one `_023.wav` per speaker, except p268/p295/p340
use `_021.wav`. Missing selected files fail explicitly. Models default to eval
mode, as in the original scripts. No trimming or additional waveform normalization is added.
Resampling overshoot is not clipped; GSV retains its original conditional
waveform scaling if the peak exceeds one.

| Model | Input rate | Embedding | Original representation |
| --- | --- | --- | --- |
| FreeVC | 16 kHz | 256 | Torch mel, partial embeddings, L2-normalized mean |
| QuickVC | 16 kHz | 256 | Log-mel, 128-frame partials, mean without re-normalization |
| TriAAN-VC | 16 kHz | 3076 | CPC plus means of final features and six skip features |
| GPT-SoVITS | 32 kHz | 512 | Magnitude STFT, first 704 bins, MelStyleEncoder |

The saved tensor dictionary uses `.squeeze().cpu()` semantics, as in the
original scripts. `--model` preserves filesystem speaker order; an explicit CSV
preserves its row order. Use a CSV matching a historical bank's key order if
exact target tie-breaking order matters.

For the historical TriAAN-VC bank, `--triaan-bank-mode train` explicitly selects
per-utterance speaker-encoder BatchNorm statistics; CPC remains in eval mode.
This is a separately labelled compatibility profile, not the original script's
eval default. No optimizer or weight training is run. BatchNorm running buffers
are updated in memory, but no model checkpoint is saved or overwritten.

Resampling uses the installed librosa default, as in the original scripts.
`--res-type` can explicitly pin a known historical resampler. The JSON receipt
records the resolved resampler, sample IDs, model-source and checkpoint hashes,
dependency versions, rate and device. Same-environment script parity does not
imply bit-identical reproduction of a bank saved under unknown older versions.

For optimization, use the [main runner](running.md) or the packaged
`create_attack_encoder` factory. It selects the original optimization modes and
adds differentiable 16-to-32-kHz resampling inside GSV; STM input remains 16 kHz.
`create_encoder` is the bank-side factory, with the eval default described above.
The bank script itself never runs an attack.

For a custom encoder factory, use `--encoder my_encoder:create_encoder` instead
of `--model`. For an explicit sample selection use `--manifest samples.csv`
instead of `--vctk-root` (columns: `speaker,utterance,wav_path`, paths relative to
the CSV). Use `--suffix _mic2.flac` for that VCTK layout. Load the saved bank with
`torch.load("artifacts/bank.pt", map_location="cpu", weights_only=True)`.

Upstream code licenses are listed in [ENCODER_LICENSES.md](../ENCODER_LICENSES.md).
Model weights and VCTK retain their separate upstream/data terms.
