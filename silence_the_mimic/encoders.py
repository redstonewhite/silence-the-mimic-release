"""Differentiable adapters for the original bank and optimization encoders.

Model definitions are imported from user-supplied upstream checkouts, except
for the small encoder-only QuickVC class. Nothing is downloaded.
See THIRD_PARTY_NOTICES.md for upstream credits and licenses.
"""

import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F


SAMPLE_RATES = {"freevc": 16000, "quickvc": 16000, "triaanvc": 16000, "gpt_sovits": 32000}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_model(name, root):
    """Avoid silently reusing an unrelated module named models/model/module."""
    top = sys.modules.get(name.split(".")[0])
    if top is not None:
        locations = list(getattr(top, "__path__", []))
        if getattr(top, "__file__", None):
            locations.append(top.__file__)
        if not locations or not all(Path(path).resolve().is_relative_to(root) for path in locations):
            raise RuntimeError(f"Module name collision for {name}; run this model in a fresh Python process")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"{name} did not load from the requested model repository")
    return module


class _FreeVC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        import torchaudio

        self.model = model
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=40,
            pad_mode="constant", norm="slaney", mel_scale="slaney",
        )

    def forward(self, wav):
        wav_slices, mel_slices = self.model.compute_partial_slices(len(wav), 1.3, 0.75)
        max_length = wav_slices[-1].stop
        if max_length >= len(wav):
            wav = F.pad(wav, (0, max_length - len(wav)), "constant", 0)
        mel = self.mel(wav).T
        partials = self.model(torch.stack([mel[part] for part in mel_slices]))
        raw = torch.mean(partials, dim=0)
        return (raw / torch.norm(raw, p=2)).squeeze()


class _QuickSpeakerEncoder(torch.nn.Module):
    """Encoder-only QuickVC model, avoiding imports of its unused synthesizer.

    Architecture/forward adapted from QuickVC models.py, Copyright (c) 2023
    quickvc (MIT). See ENCODER_LICENSES.md. Parameter names are unchanged.
    """

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(80, 256, 3, batch_first=True)
        self.linear = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        raw = self.relu(self.linear(hidden[-1]))
        return raw / torch.norm(raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        return [torch.arange(index, index + partial_frames)
                for index in range(0, total_frames - partial_frames, partial_hop)]


class _QuickVC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        from librosa.filters import mel

        self.model = model
        self.register_buffer("mel_basis", torch.from_numpy(mel(
            sr=16000, n_fft=1280, n_mels=80, fmin=0.0, fmax=None,
        )))
        self.register_buffer("window", torch.hann_window(1280))

    def forward(self, wav):
        y = F.pad(wav.unsqueeze(0).unsqueeze(1), (480, 480), mode="reflect").squeeze(1)
        spec = torch.stft(
            y, 1280, hop_length=320, win_length=1280, window=self.window,
            center=False, pad_mode="reflect", normalized=False,
            onesided=True, return_complex=True,
        )
        mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, torch.abs(spec)), min=1e-5))
        mel = mel.transpose(1, 2)
        last = mel[:, -128:]
        if mel.size(1) > 128:
            slices = self.model.compute_partial_slices(mel.size(1), 128, 64)
            mels = [mel[:, part] for part in slices]
            mels.append(last)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
            embedding = torch.mean(self.model(mels), axis=0).unsqueeze(0)
        else:
            embedding = self.model(last)
        # The historical QuickVC adapter does not re-normalize this mean.
        return embedding.squeeze()


class _TriAANVC(torch.nn.Module):
    def __init__(self, model, cpc):
        super().__init__()
        self.model = model
        self.cpc = cpc

    def forward(self, wav):
        features = self.cpc(wav.unsqueeze(0).unsqueeze(0), None)[0].transpose(1, 2)
        encoded, skips = self.model(features)
        return torch.cat([encoded.mean(dim=2).flatten()] + [skip.mean(dim=2).flatten() for skip in skips])


class _GSV(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wav):
        maximum = wav.abs().max()
        if maximum > 1:
            wav = wav / min(2, maximum)
        y = F.pad(wav.unsqueeze(0).unsqueeze(1), (704, 704), mode="reflect").squeeze(1)
        spec = torch.stft(
            y, 2048, hop_length=640, win_length=2048,
            window=torch.hann_window(2048, device=wav.device), center=False,
            pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
        )
        return self.model(torch.abs(spec)[:, :704]).squeeze()


def create_encoder(name, *, model_root, checkpoint, device="cpu", cpc_checkpoint=None,
                   triaan_bank_mode="eval"):
    """Load the original model-specific representation, in frozen eval mode.

    FreeVC takes its native speaker checkpoint; the other models take extracted
    speaker-encoder state dictionaries, not a full synthesizer checkpoint.
    Use create_attack_encoder for the original 16-kHz optimization settings.
    """
    if name not in SAMPLE_RATES:
        raise ValueError(f"Unknown model: {name}")
    if triaan_bank_mode not in {"eval", "train"}:
        raise ValueError("triaan_bank_mode must be eval or train")
    if name != "triaanvc" and triaan_bank_mode != "eval":
        raise ValueError("The bank-mode compatibility option is only for TriAAN-VC")
    root = Path(model_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Provide an existing model repository and encoder checkpoint")
    if name != "triaanvc" and cpc_checkpoint is not None:
        raise ValueError("cpc_checkpoint is only used with triaanvc")
    sources = []
    if name == "freevc":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get("model_state"), dict) or not state["model_state"]:
            raise ValueError("FreeVC checkpoint must contain a nonempty model_state dictionary")
        module = _import_model("speaker_encoder.voice_encoder", root)
        model = module.SpeakerEncoder(checkpoint, device=device, verbose=False)
        # The upstream constructor loads with strict=False. Require every
        # inference parameter; allow only the known GE2E training-loss scalars
        # absent from the native inference model, never arbitrary extra keys.
        expected = model.state_dict()
        inference_state = {key: value for key, value in state["model_state"].items()
                           if key in expected or key not in {"similarity_weight", "similarity_bias"}}
        model.load_state_dict(inference_state, strict=True)
        encoder = _FreeVC(model)
        sources += [root / "speaker_encoder/voice_encoder.py", root / "speaker_encoder/hparams.py"]
    elif name == "quickvc":
        model = _QuickSpeakerEncoder()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        encoder = _QuickVC(model)
        sources.append(root / "models.py")
    elif name == "triaanvc":
        if cpc_checkpoint is None or not Path(cpc_checkpoint).is_file():
            raise FileNotFoundError("TriAAN-VC also requires its original CPC checkpoint")
        cpc_checkpoint = Path(cpc_checkpoint).resolve()
        module = _import_model("model.model", root)
        model = module.SpeakerEncoder(c_in=256, c_out=4, num_layer=6, c_h=512)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        cpc_path = root / "src/cpc.py"
        spec = importlib.util.spec_from_file_location("_stm_triaan_cpc", cpc_path)
        cpc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpc_module)
        config = cpc_module.get_default_cpc_config()
        cpc = cpc_module.CPCModel(cpc_module.getEncoder(config), cpc_module.getAR(config))
        cpc.load_state_dict(torch.load(cpc_checkpoint, map_location="cpu", weights_only=True)["weights"], strict=True)
        encoder = _TriAANVC(model, cpc)
        sources += [root / "model/model.py", root / "model/attention.py", root / "model/conv_modules.py", cpc_path]
    else:
        import_root = root / "GPT_SoVITS" if (root / "GPT_SoVITS").is_dir() else root
        module = _import_model("module.modules", import_root)
        model = module.MelStyleEncoder(n_mel_channels=704, style_vector_dim=512)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("spectral.0.fc.weight", torch.empty(0)).shape != torch.Size([128, 704]):
            raise ValueError("The original GSV bank requires the 704-bin MelStyleEncoder checkpoint; do not substitute a 1025-bin encoder")
        model.load_state_dict(state, strict=True)
        encoder = _GSV(model)
        sources.append(import_root / "module/modules.py")
    encoder.to(device).eval().requires_grad_(False)
    if name == "triaanvc" and triaan_bank_mode == "train":
        # Archive compatibility: per-utterance BatchNorm statistics, without
        # gradients/optimization. CPC stays in eval; checkpoint is not modified.
        encoder.model.train()
    encoder.sample_rate = SAMPLE_RATES[name]
    encoder.provenance = {
        "model": name, "model_root": str(root), "mode": "eval", "embedding_squeeze": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sources},
    }
    if (root / ".git").exists():
        encoder.provenance["model_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
    if cpc_checkpoint is not None:
        encoder.provenance.update(cpc_checkpoint=str(cpc_checkpoint), cpc_sha256=sha256_file(cpc_checkpoint))
        encoder.provenance.update(triaan_bank_mode=triaan_bank_mode,
                                  mode="speaker_train_cpc_eval" if triaan_bank_mode == "train" else "eval")
    return encoder


class _GSV16k(torch.nn.Module):
    """Original attack-side resampling; the bank frontend still takes 32 kHz."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform):
        import torchaudio

        return self.encoder(torchaudio.functional.resample(waveform.squeeze(0), 16000, 32000))


def create_attack_encoder(name, *, model_root, checkpoint, device="cpu", cpc_checkpoint=None):
    """Load the frozen, 16-kHz waveform encoder used by the original STM runner.

    FreeVC/QuickVC/TriAAN-VC retain the original optimization-time train mode
    (including TriAAN's CPC); GSV stays in eval mode and resamples internally.
    Train mode changes layer behavior, not weights: all parameters are frozen.
    No model checkpoint is written or downloaded.
    """
    encoder = create_encoder(name, model_root=model_root, checkpoint=checkpoint,
                             device=device, cpc_checkpoint=cpc_checkpoint)
    provenance = dict(encoder.provenance)
    if name == "gpt_sovits":
        encoder = _GSV16k(encoder).to(device).eval().requires_grad_(False)
        provenance.update(mode="eval", internal_resampling="torchaudio.functional.resample:16000->32000")
    else:
        encoder.train().requires_grad_(False)
        provenance.update(mode="train", internal_resampling=None)
    encoder.sample_rate = 16000
    provenance.update(input_sample_rate=16000, bank_input_sample_rate=SAMPLE_RATES[name],
                      optimization_weights_frozen=True)
    encoder.provenance = provenance
    return encoder
