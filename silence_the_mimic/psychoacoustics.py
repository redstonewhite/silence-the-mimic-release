"""Bark masking thresholds used by the original STM implementation.

Adapted from Python-Audio-Coder.
See THIRD_PARTY_NOTICES.md for attribution and licensing.
The original frame loop, float32 mapping matrices and Nyquist-bin treatment
are retained for numerical compatibility. Threshold construction runs on CPU.
"""

import torch


def _hz_to_bark(frequency):
    if not torch.is_tensor(frequency):
        frequency = torch.tensor(frequency)
    return 6.0 * torch.arcsinh(frequency / 600.0)


def _bark_to_hz(bark):
    return 600.0 * torch.sinh(bark / 6.0)


def _spreading_matrix(max_frequency, n_bands, alpha):
    max_bark = _hz_to_bark(max_frequency)
    prototype_db = torch.zeros(2 * n_bands)
    prototype_db[:n_bands] = torch.linspace(-max_bark * 27, -8, n_bands) - 23.5
    prototype_db[n_bands:] = (
        torch.linspace(0, -max_bark * 12.0, n_bands) - 23.5
    )
    prototype = 10.0 ** (prototype_db / 20.0 * alpha)
    spreading = torch.zeros((n_bands, n_bands))
    for band in range(n_bands):
        spreading[band, :] = prototype[n_bands - band : 2 * n_bands - band]
    return spreading


def _bark_mapping(sample_rate, n_bands, n_fft):
    max_bark = _hz_to_bark(sample_rate / 2)
    step = max_bark / (n_bands - 1)
    bin_bark = _hz_to_bark(
        torch.linspace(0, n_fft / 2, n_fft // 2 + 1) * sample_rate / n_fft
    )
    mapping = torch.zeros((n_bands, n_fft))
    for band in range(n_bands):
        mapping[band, : n_fft // 2 + 1] = torch.round(bin_bark / step) == band
    inverse = torch.matmul(
        torch.diag((1.0 / (torch.sum(mapping, 1) + 1e-6)) ** 0.5),
        mapping[:, : n_fft // 2 + 1],
    ).T
    return mapping, inverse


def _threshold_in_bark(magnitude, spreading, alpha, sample_rate, n_bands):
    threshold = torch.matmul(magnitude**alpha, spreading**alpha)
    threshold = threshold ** (1.0 / alpha)
    max_bark = _hz_to_bark(sample_rate / 2.0)
    barks = torch.arange(n_bands) * (max_bark / (n_bands - 1))
    frequency = _bark_to_hz(barks) + 1e-6
    threshold_in_quiet = torch.clip(
        3.64 * (frequency / 1000.0) ** -0.8
        - 6.5 * torch.exp(-0.6 * (frequency / 1000.0 - 3.3) ** 2.0)
        + 1e-3 * ((frequency / 1000.0) ** 4.0),
        -20,
        120,
    )
    return torch.max(threshold, 10.0 ** ((threshold_in_quiet - 60) / 20))


@torch.no_grad()
def masking_threshold(spectra: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Return linear-amplitude thresholds for a scaled, one-sided CPU STFT.

    ``spectra`` has shape [frequency bins, time frames]. The caller supplies
    the same scaled STFT used by legacy STM (waveform multiplied by 10,000).
    This numerical scale affects the absolute threshold in quiet.
    """
    if spectra.device.type != "cpu":
        raise ValueError("Masking-threshold preparation expects CPU spectra")
    if spectra.ndim != 2 or not spectra.is_complex() or spectra.shape[0] < 2:
        raise ValueError("Expected a complex [frequency, time] STFT")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    alpha, n_bands = 0.8, 64
    n_frequencies = spectra.shape[0] - 1
    n_fft = 2 * n_frequencies
    mapping, inverse = _bark_mapping(sample_rate, n_bands, n_fft)
    spreading = _spreading_matrix(sample_rate / 2, n_bands, alpha)
    thresholds = torch.zeros(spectra.shape)
    for frame in range(spectra.shape[1]):
        magnitude = torch.abs(spectra[:, frame])
        bark_magnitude = torch.matmul(
            magnitude[:n_frequencies] ** 2.0,
            mapping[:, :n_frequencies].T,
        ) ** 0.5
        bark_threshold = _threshold_in_bark(
            bark_magnitude, spreading, alpha, sample_rate, n_bands
        )
        thresholds[:, frame] = torch.matmul(
            bark_threshold, inverse[:, :n_frequencies].T.float()
        )
    return thresholds
