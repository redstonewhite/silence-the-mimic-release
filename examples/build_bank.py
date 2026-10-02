"""Build a referral bank using an original-model adapter or encoder factory.

The factory accepts ``device`` and returns the waveform-to-embedding callable
used with STM. It owns checkpoint loading and encoder train/eval mode.
One explicitly selected utterance is encoded per speaker, without additional
embedding averaging or normalization. No checkpoints or cloning systems are bundled.
"""

import argparse
import csv
import importlib
import inspect
import json
from pathlib import Path
import random

import numpy as np
import torch


VCTK_EXCEPTIONS = {"p268", "p295", "p340"}


def validate_records(records):
    if not records:
        raise ValueError("No bank utterances selected")
    seen = set()
    for row in records:
        if not row["speaker"] or not row["utterance"]:
            raise ValueError("Speaker and utterance IDs must be nonempty")
        if row["speaker"] in seen:
            raise ValueError(f"Duplicate bank speaker: {row['speaker']}")
        seen.add(row["speaker"])
        if not Path(row["wav_path"]).is_file():
            raise FileNotFoundError(f"{row['speaker']}/{row['utterance']}: {row['wav_path']}")
    return records


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"speaker", "utterance", "wav_path"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain speaker,utterance,wav_path columns")
        for row in reader:
            if any(not (row.get(key) or "").strip() for key in ("speaker", "utterance", "wav_path")):
                raise ValueError(f"Incomplete manifest row at line {reader.line_num}")
            audio_path = Path(row["wav_path"].strip())
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            records.append({
                "speaker": row["speaker"].strip(),
                "utterance": row["utterance"].strip(),
                "wav_path": str(audio_path.resolve()),
            })
    return validate_records(records)


def vctk_records(root, suffix=".wav", *, sort=True):
    """Fixed historical selection, not a search for the longest recording."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not suffix or "/" in suffix or "\\" in suffix:
        raise ValueError("suffix must be a filename suffix, e.g. .wav or _mic2.flac")
    records = []
    folders = sorted(root.iterdir()) if sort else root.iterdir()
    for folder in folders:
        if folder.is_dir() and folder.name.startswith("p") and folder.name[1:].isdigit():
            number = "021" if folder.name in VCTK_EXCEPTIONS else "023"
            utterance = f"{folder.name}_{number}"
            records.append({
                "speaker": folder.name, "utterance": utterance,
                "wav_path": str(folder / f"{utterance}{suffix}"),
            })
    return validate_records(records)


def build_bank(encode, records, *, sample_rate=16000, device="cpu", res_type=None):
    import librosa

    if not callable(encode):
        raise TypeError("Encoder factory must return a callable")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    records = validate_records(records)
    bank, shape, dtype = {}, None, None
    for row in records:
        try:
            options = {"res_type": res_type} if res_type is not None else {}
            waveform, _ = librosa.load(
                row["wav_path"], sr=sample_rate, mono=True,
                dtype=np.float32, **options,
            )
            # Resampling can slightly overshoot PCM [-1, 1]. The original bank
            # scripts do not clip or normalize it; keep that input unchanged.
            if not waveform.size or not np.isfinite(waveform).all():
                raise ValueError("Expected nonempty, finite audio")
            with torch.no_grad():
                embedding = encode(torch.from_numpy(waveform).to(device))
            if not isinstance(embedding, torch.Tensor) or not embedding.numel():
                raise ValueError("Encoder must return a nonempty Tensor")
            if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
                raise ValueError("Encoder returned an invalid embedding")
            if shape is None:
                shape, dtype = embedding.shape, embedding.dtype
            if embedding.shape != shape or embedding.dtype != dtype:
                raise ValueError("All bank embeddings must have matching shapes and dtypes")
            bank[row["speaker"]] = embedding.detach().cpu().clone()
        except Exception as error:
            raise ValueError(f"Failed bank sample {row['speaker']}/{row['utterance']}: {error}") from error
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV: speaker,utterance,wav_path")
    source.add_argument("--vctk-root", type=Path, help="Directory containing VCTK speaker folders")
    parser.add_argument("--suffix", default=".wav", help="VCTK filename suffix; e.g. _mic2.flac")
    encoder_choice = parser.add_mutually_exclusive_group(required=True)
    encoder_choice.add_argument("--encoder", help="Factory as module:function; receives device")
    encoder_choice.add_argument("--model", choices=("freevc", "quickvc", "triaanvc", "gpt_sovits"))
    parser.add_argument("--model-root", type=Path, help="Upstream model checkout, for --model")
    parser.add_argument("--checkpoint", type=Path, help="Original speaker-encoder weights, for --model")
    parser.add_argument("--cpc-checkpoint", type=Path, help="Required additionally for TriAAN-VC")
    parser.add_argument("--triaan-bank-mode", choices=("eval", "train"),
                        help="Explicit TriAAN-VC archive compatibility; default eval follows the original script")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, help="Default: 32000 for original GSV; 16000 otherwise")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--res-type", help="Omit to follow the installed librosa default, as in the original scripts")
    parser.add_argument("--speaker-order", choices=("filesystem", "sorted"), help="Default: filesystem for --model; sorted for --encoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Refusing to overwrite an existing bank or build receipt")
    if args.model:
        from silence_the_mimic.encoders import SAMPLE_RATES, create_encoder
        if args.model_root is None or args.checkpoint is None:
            parser.error("--model requires --model-root and --checkpoint")
        if args.triaan_bank_mode is not None and args.model != "triaanvc":
            parser.error("--triaan-bank-mode is only for --model triaanvc")
        expected_rate = SAMPLE_RATES[args.model]
        if args.sample_rate is not None and args.sample_rate != expected_rate:
            parser.error(f"The original {args.model} bank uses {expected_rate} Hz")
        sample_rate = expected_rate
    else:
        if any(value is not None for value in (args.model_root, args.checkpoint, args.cpc_checkpoint, args.triaan_bank_mode)):
            parser.error("Model checkpoint arguments require --model")
        sample_rate = 16000 if args.sample_rate is None else args.sample_rate
    if sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    order = args.speaker_order or ("filesystem" if args.model else "sorted")
    records = read_manifest(args.manifest) if args.manifest else vctk_records(args.vctk_root, args.suffix, sort=order == "sorted")
    if len(records) < 6:
        parser.error("The default STM target selection requires at least six speakers")
    try:
        import librosa
    except ImportError as error:
        raise SystemExit('Install bank-building dependencies with: python -m pip install ".[audio]"') from error
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model:
        encode = create_encoder(args.model, model_root=args.model_root, checkpoint=args.checkpoint,
                                device=args.device, cpc_checkpoint=args.cpc_checkpoint,
                                triaan_bank_mode=args.triaan_bank_mode or "eval")
    else:
        module, separator, attribute = args.encoder.partition(":")
        if not separator or not module or not attribute:
            parser.error("--encoder must have the form module:function")
        encode = getattr(importlib.import_module(module), attribute)(device=torch.device(args.device))
    bank = build_bank(encode, records, sample_rate=sample_rate, device=args.device, res_type=args.res_type)
    metadata = {
        "encoder_factory": args.encoder, "seed": args.seed, "device": args.device,
        "model": getattr(encode, "provenance", None),
        "speaker_order": "manifest" if args.manifest else order,
        "selection": "explicit_manifest" if args.manifest else "VCTK_023_with_021_exceptions",
        "preprocessing": {"sample_rate": sample_rate, "mono": True,
                          "res_type": args.res_type or inspect.signature(librosa.load).parameters["res_type"].default,
                          "trim": False, "waveform_normalization": False},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "librosa": librosa.__version__},
        "embedding_shape": list(next(iter(bank.values())).shape),
        "embedding_dtype": str(next(iter(bank.values())).dtype),
        "samples": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with receipt.open("x", encoding="utf-8") as receipt_handle:
            created.append(receipt)
            with output.open("xb") as bank_handle:
                created.append(output)
                torch.save(bank, bank_handle)
            json.dump(metadata, receipt_handle, indent=2)
            receipt_handle.write("\n")
    except BaseException:
        # Remove only incomplete files created here, never preexisting outputs.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(f"Saved {len(bank)} speakers to {output}; build receipt: {receipt}")


if __name__ == "__main__":
    main()
