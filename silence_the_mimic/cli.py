"""Standalone audio runner; core optimization remains in core.py."""

import argparse
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .core import STM, STMConfig
from .encoders import SAMPLE_RATES, create_attack_encoder, sha256_file


def dependency_versions():
    versions = {"torch": str(torch.__version__), "numpy": np.__version__}
    for name in ("librosa", "torchaudio", "soundfile"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def plan_outputs(source, destination):
    """Resolve the complete batch and reject existing outputs before loading models."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_file():
        if destination.suffix.lower() != ".wav":
            raise ValueError("Single-file --output must be a .wav path")
        pairs = [(source, destination)]
    elif source.is_dir():
        if destination.is_relative_to(source) or destination.is_file():
            raise ValueError("Directory --output must be a separate directory outside --input")
        pairs = [(path, destination / path.relative_to(source)) for path in sorted(source.rglob("*.wav"))]
        if not pairs:
            raise ValueError("Input directory contains no .wav files")
    else:
        raise FileNotFoundError(source)
    for _, output in pairs:
        for path in (output, output.with_suffix(".wav.json")):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    return pairs


def load_waveform(path):
    import soundfile as sf

    waveform, rate = sf.read(path, dtype="float32")
    if rate != 16000 or waveform.ndim != 1:
        raise ValueError("Input must already be mono, 16 kHz; no implicit resampling/downmixing")
    if not waveform.size or not np.isfinite(waveform).all() or np.abs(waveform).max() > 1:
        raise ValueError("Input must be nonempty, finite float32 audio within [-1, 1]")
    return np.ascontiguousarray(waveform)


def check_bank_receipt(bank_path, model, checkpoint, cpc_checkpoint=None):
    """Check model metadata if present; an older tensor-only bank is unverified."""
    receipt = Path(bank_path).with_suffix(Path(bank_path).suffix + ".json")
    if not receipt.exists():
        return None
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    provenance = metadata.get("model")
    if provenance is None:
        return None
    if provenance.get("model") != model:
        raise ValueError("Bank receipt specifies a different model")
    if provenance.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("Bank receipt specifies a different encoder checkpoint")
    if metadata.get("preprocessing", {}).get("sample_rate") != SAMPLE_RATES[model]:
        raise ValueError("Bank receipt specifies a different bank input sample rate")
    if model == "triaanvc" and (cpc_checkpoint is None or provenance.get("cpc_sha256") != sha256_file(cpc_checkpoint)):
        raise ValueError("Bank receipt specifies a different CPC checkpoint")
    return metadata


def save_result(result, destination, metadata, subtype="FLOAT"):
    """Write waveform and receipt exclusively, without normalizing or clipping."""
    import soundfile as sf

    destination = Path(destination)
    receipt = destination.with_suffix(".wav.json")
    waveform = result.waveform.detach().cpu().numpy()
    if subtype not in {"FLOAT", "PCM_16"}:
        raise ValueError("Output subtype must be FLOAT or PCM_16")
    if not np.isfinite(waveform).all():
        raise ValueError("Output waveform is not finite")
    if subtype == "PCM_16" and np.abs(waveform).max() > 1:
        raise ValueError("PCM_16 would clip this output; use FLOAT instead")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with receipt.open("x", encoding="utf-8") as receipt_handle:
            created.append(receipt)
            with destination.open("xb") as audio_handle:
                created.append(destination)
                sf.write(audio_handle, waveform, 16000, format="WAV", subtype=subtype)
            metadata = dict(metadata, output_sha256=sha256_file(destination),
                            output_peak=float(np.abs(waveform).max()), output_subtype=subtype,
                            output_normalization=False, output_clipping=False)
            json.dump(metadata, receipt_handle, indent=2)
            receipt_handle.write("\n")
    except BaseException:
        # Only remove incomplete files created by this call, never prior outputs.
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Protect preprocessed mono 16-kHz audio with STM")
    parser.add_argument("--model", choices=tuple(SAMPLE_RATES), required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Speaker encoder, not a full synthesizer checkpoint")
    parser.add_argument("--cpc-checkpoint", help="Required for TriAAN-VC")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--input", required=True, help="Audio file or directory of .wav files")
    parser.add_argument("--output", required=True, help="New .wav file or separate output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trim-top-db", type=float, help="Explicit optional librosa trimming; default: no trim")
    parser.add_argument("--output-subtype", choices=("FLOAT", "PCM_16"), default="FLOAT")
    args = parser.parse_args(argv)
    if (args.model == "triaanvc") != (args.cpc_checkpoint is not None):
        parser.error("--cpc-checkpoint is required for, and only used with, triaanvc")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.trim_top_db is not None and (not np.isfinite(args.trim_top_db) or args.trim_top_db < 0):
        parser.error("--trim-top-db must be finite and nonnegative")
    config = STMConfig(iterations=args.iterations)
    pairs = plan_outputs(args.input, args.output)
    if args.seed + len(pairs) - 1 >= 2**32:
        parser.error("Batch per-file seeds would exceed 2**32 - 1")
    device = torch.device(args.device)
    bank_path = Path(args.bank).resolve()
    bank = torch.load(bank_path, map_location=device, weights_only=True)
    if not isinstance(bank, dict) or len(bank) < config.target_rank:
        raise ValueError(f"Bank must be a tensor dictionary with at least {config.target_rank} speakers")
    bank_receipt = check_bank_receipt(bank_path, args.model, args.checkpoint, args.cpc_checkpoint)
    if bank_receipt is None:
        print("Warning: bank has no model receipt; encoder/preprocessing compatibility is unverified", file=sys.stderr)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder = create_attack_encoder(args.model, model_root=args.model_root, checkpoint=args.checkpoint,
                                    device=device, cpc_checkpoint=args.cpc_checkpoint)
    if bank_receipt is not None and bank_receipt["model"].get("source_sha256") != encoder.provenance["source_sha256"]:
        raise ValueError("Bank receipt specifies different upstream encoder source files")
    stm = STM(encoder, config, device=device)
    bank_hash = sha256_file(bank_path)
    for index, (source, destination) in enumerate(pairs):
        try:
            waveform = load_waveform(source)
            trim_indices = None
            if args.trim_top_db is not None:
                import librosa

                waveform, trim_indices = librosa.effects.trim(waveform, top_db=args.trim_top_db)
                trim_indices = trim_indices.tolist()
            seed = args.seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = stm.protect(waveform, sample_rate=16000, referral_bank=bank, seed=seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - start
            metadata = {
                "status": "ok", "input": str(source), "input_sha256": sha256_file(source),
                "output": str(destination), "bank": str(bank_path), "bank_sha256": bank_hash,
                "bank_encoder_metadata_checked": bank_receipt is not None,
                "encoder": encoder.provenance, "config": asdict(config), "seed": seed,
                "seed_rule": "base seed + index in sorted relative input paths",
                "sample_rate": 16000, "input_samples_after_trim": len(waveform),
                "trim_top_db": args.trim_top_db, "trim_indices": trim_indices,
                "target_speaker": result.target_speaker, "protect_wall_seconds": wall_seconds,
                "timing_scope": "synchronized STM.protect call; no warmup; not a formal latency benchmark",
                "versions": dependency_versions(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "determinism": {"algorithms": torch.are_deterministic_algorithms_enabled(),
                                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                "command": [sys.executable, "-m", "silence_the_mimic", *(sys.argv[1:] if argv is None else argv)],
            }
            save_result(result, destination, metadata, args.output_subtype)
            print(f"Saved {source.name} -> {destination}; target={result.target_speaker}; {wall_seconds:.3f} s")
        except Exception as error:
            raise RuntimeError(f"Failed input {source}; stopped without skipping it: {error}") from error


if __name__ == "__main__":
    main()
