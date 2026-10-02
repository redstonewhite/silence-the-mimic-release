"""STM in log-magnitude coordinates, with fixed phase and hard projection."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .psychoacoustics import masking_threshold


@dataclass(frozen=True)
class STMConfig:
    """Defaults from run_stm.py; amplitude coordinates use log10, not 20*log10."""

    iterations: int = 80
    learning_rate: float = 1.0
    lr_gamma: float = 0.9
    n_fft: int = 512
    hop_length: int = 256
    scale_factor: float = 10000.0
    log_epsilon: float = 1e-5
    noise_std: float = 1e-3
    inaudible_headroom_multiplier: float = 1.0
    inaudible_headroom_shift: float = 0.0
    inaudible_floor: float = -4.0
    audible_headroom: float = 0.15
    audible_floor: float = -0.15
    audible_weight: float = 1.0
    tv_weight: float = 0.2
    tv_lags: tuple[int, ...] = (1, 2, 3, 4)
    target_rank: int = 6

    def __post_init__(self):
        for name in ("iterations", "n_fft", "hop_length", "target_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_fft < 4 or self.n_fft % 2 or self.hop_length > self.n_fft:
            raise ValueError("Use an even n_fft >= 4 and hop_length <= n_fft")
        for name in ("learning_rate", "lr_gamma", "scale_factor", "log_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "noise_std", "inaudible_headroom_multiplier", "tv_weight", "audible_weight"
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "inaudible_headroom_shift", "inaudible_floor", "audible_headroom", "audible_floor"
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.tv_lags or any(
            isinstance(lag, bool) or not isinstance(lag, int) or lag < 1
            for lag in self.tv_lags
        ):
            raise ValueError("tv_lags must contain positive integers")


@dataclass(frozen=True)
class STMResult:
    """Detached tensors on the selected device, after all requested updates."""

    waveform: torch.Tensor
    perturbation: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    target_speaker: str | None


@dataclass
class _Spectrum:
    log_magnitude: torch.Tensor
    phase: torch.Tensor
    weights: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    window: torch.Tensor
    length: int


class STM:
    """Protect one preprocessed utterance using a differentiable encoder.

    ``speaker_encoder`` maps a one-dimensional waveform on ``device`` to a
    nonempty embedding Tensor, and must preserve gradients to the waveform.
    Model loading, input decoding/resampling/trimming and output encoding are
    the caller's responsibility. The caller also controls encoder train/eval
    mode and deterministic settings; STM never changes those modes.
    """

    def __init__(
        self,
        speaker_encoder: Callable[[torch.Tensor], torch.Tensor],
        config: STMConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ):
        if not callable(speaker_encoder):
            raise TypeError("speaker_encoder must be callable")
        self.encoder = speaker_encoder
        self.config = config or STMConfig()
        self.device = torch.device(device)

    def _embedding(self, waveform):
        embedding = self.encoder(waveform)
        if not isinstance(embedding, torch.Tensor) or embedding.numel() == 0:
            raise ValueError("speaker_encoder must return a nonempty Tensor")
        if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
            raise ValueError("speaker_encoder returned an invalid embedding")
        if embedding.device != waveform.device:
            raise ValueError("Embedding and waveform must be on the same device")
        return embedding

    def _prepare(self, waveform: np.ndarray, sample_rate: int):
        cfg = self.config
        scaled = np.clip(
            waveform * cfg.scale_factor, -cfg.scale_factor, cfg.scale_factor
        )
        window = torch.hann_window(cfg.n_fft)
        spectra = torch.stft(
            torch.from_numpy(scaled), n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, window=window,
            normalized=True, return_complex=True,
        )
        threshold = masking_threshold(spectra, sample_rate) / cfg.scale_factor
        magnitude = torch.abs(spectra).detach() / cfg.scale_factor
        inaudible = (magnitude < threshold).int()
        audible = (magnitude >= threshold).int()
        weights = (
            inaudible + torch.ones_like(inaudible).int() * cfg.audible_weight * audible
        )
        log_magnitude = torch.log10(magnitude + cfg.log_epsilon)
        log_threshold = torch.log10(threshold + cfg.log_epsilon)
        upper = (
            (torch.max(log_threshold - log_magnitude, torch.tensor(0.0))
             + cfg.inaudible_headroom_shift) * cfg.inaudible_headroom_multiplier
            + audible * cfg.audible_headroom
        )
        lower = cfg.inaudible_floor * inaudible + cfg.audible_floor * audible
        if not torch.isfinite(upper).all() or not torch.isfinite(lower).all():
            raise ValueError("Nonfinite psychoacoustic bounds")
        if (lower > upper).any():
            raise ValueError("Configuration produces an empty feasible interval")
        if cfg.tv_weight > 0 and spectra.shape[1] <= max(cfg.tv_lags):
            raise ValueError("Utterance is too short for the configured TV lags")
        return _Spectrum(
            log_magnitude.to(self.device), torch.angle(spectra).detach().to(self.device),
            weights.to(self.device), lower.to(self.device), upper.to(self.device),
            window.to(self.device), len(waveform),
        )

    def _reconstruct(self, state: _Spectrum, delta: torch.Tensor):
        magnitude = torch.pow(10, state.log_magnitude + delta * state.weights)
        spectra = magnitude * torch.exp(1j * state.phase)
        return torch.istft(
            spectra, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            window=state.window, normalized=True, return_complex=False,
            length=state.length,
        )

    def _target(self, original, bank):
        if len(bank) < self.config.target_rank:
            raise ValueError(f"Referral bank requires >= {self.config.target_rank} speakers")
        distances, candidates = [], {}
        for speaker, value in bank.items():
            if not isinstance(speaker, str):
                raise TypeError("Referral-bank speaker IDs must be strings")
            candidate = torch.as_tensor(value, device=self.device).detach()
            if candidate.shape != original.shape or not torch.isfinite(candidate).all():
                raise ValueError(f"Invalid referral embedding for {speaker}")
            candidates[speaker] = candidate
            distances.append((speaker, F.mse_loss(candidate, original).item()))
        # Python's stable sort preserves insertion order in exact ties, as in STM.
        speaker = sorted(distances, key=lambda item: item[1], reverse=True)[
            self.config.target_rank - 1
        ][0]
        return candidates[speaker], speaker

    def protect(
        self,
        waveform: np.ndarray | torch.Tensor,
        *,
        sample_rate: int,
        referral_bank: Mapping[str, torch.Tensor] | None = None,
        target_embedding: torch.Tensor | None = None,
        seed: int = 42,
    ) -> STMResult:
        """Run STM with a bank-selected target or one explicitly supplied target.

        Inputs are mono float32 samples in [-1, 1], already at the
        encoder's expected sample rate. There is no implicit trim, loudness
        normalization or resampling. Initialization uses a private CPU RNG.
        """
        if (referral_bank is None) == (target_embedding is None):
            raise ValueError("Supply exactly one of referral_bank or target_embedding")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform)
        if waveform.ndim != 1 or waveform.dtype != np.dtype("float32"):
            raise ValueError("Expected a mono float32 waveform")
        if len(waveform) <= self.config.n_fft // 2:
            raise ValueError("Utterance is too short for STFT reflection padding")
        if not np.isfinite(waveform).all() or np.abs(waveform).max() > 1:
            raise ValueError("Waveform must be finite and within [-1, 1]")
        waveform = np.ascontiguousarray(waveform)
        state = self._prepare(waveform, sample_rate)
        original_waveform = torch.from_numpy(waveform).to(self.device)
        with torch.no_grad():
            original = self._embedding(original_waveform)
            if referral_bank is not None:
                target, target_speaker = self._target(original, referral_bank)
            else:
                target = torch.as_tensor(target_embedding, device=self.device).detach()
                target_speaker = None
                if target.shape != original.shape or not torch.isfinite(target).all():
                    raise ValueError("target_embedding must match the encoder's output shape")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        # randn_like in legacy STM preserves the STFT's frequency-major strides.
        # A plain randn(shape) assigns the same random draws to different bins.
        initial_noise = torch.empty_like(state.log_magnitude, device="cpu")
        initial_noise.normal_(generator=generator)
        delta = (
            (initial_noise * self.config.noise_std).to(self.device).requires_grad_(True)
        )
        optimizer = torch.optim.Adam([delta], lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config.lr_gamma
        )
        with torch.enable_grad():
            for _ in range(self.config.iterations):
                optimizer.zero_grad()
                embedding = self._embedding(self._reconstruct(state, delta))
                if not embedding.requires_grad:
                    raise ValueError("speaker_encoder detached the waveform gradient")
                cosine = F.cosine_similarity(embedding, target, dim=-1).mean()
                l1_distance = F.l1_loss(embedding, target)
                tv_terms = []
                if self.config.tv_weight > 0:
                    tv_terms = [
                        torch.mean(torch.abs(delta[:, lag:] - delta[:, :-lag]))
                        for lag in self.config.tv_lags
                    ]
                # Keep the legacy graph-construction and addition order: it also
                # determines float32 rounding when branch gradients accumulate.
                loss = l1_distance + (1 - cosine)
                for term in tv_terms:
                    loss = loss + self.config.tv_weight * term
                if not torch.isfinite(loss):
                    raise ValueError("STM loss became nonfinite")
                # Differentiate only the perturbation; leave encoder .grad untouched.
                delta.grad, = torch.autograd.grad(loss, delta)
                if not torch.isfinite(delta.grad).all():
                    raise ValueError("STM gradient became nonfinite")
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    delta.copy_(torch.clamp(delta, state.lower, state.upper))

        # Reconstruct after the final update AND projection (all 80 updates).
        with torch.no_grad():
            protected = self._reconstruct(state, delta)
        return STMResult(
            protected.detach(), delta.detach(), state.lower, state.upper, target_speaker
        )
