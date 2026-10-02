"""CPU smoke example with a toy encoder; no protection-performance claim."""

import torch
import torch.nn.functional as F

from silence_the_mimic import STM, STMConfig


def toy_encoder(waveform):
    features = torch.stack([
        waveform.mean(), waveform.square().mean(),
        (waveform[1:] - waveform[:-1]).square().mean(),
        waveform.abs().mean(),
    ])
    return F.normalize(features, dim=0)


def create_encoder(device):
    """Toy factory for testing the bank builder; no protection claim."""
    return toy_encoder


def main():
    sample_rate = 16000
    time = torch.arange(4096) / sample_rate
    waveform = 0.2 * torch.sin(2 * torch.pi * 220 * time)
    bank = {
        f"speaker_{index}": F.normalize(torch.tensor([1.0, index + 1.0, 0.1, 0.5]), dim=0)
        for index in range(6)
    }
    result = STM(toy_encoder, STMConfig(iterations=4)).protect(
        waveform, sample_rate=sample_rate, referral_bank=bank,
    )
    print(f"Target: {result.target_speaker}; output samples: {result.waveform.numel()}")
    print("Finite output:", bool(torch.isfinite(result.waveform).all()))
    print("Within bounds:", bool(
        ((result.perturbation >= result.lower_bound)
         & (result.perturbation <= result.upper_bound)).all()
    ))


if __name__ == "__main__":
    main()
