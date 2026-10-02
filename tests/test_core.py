"""Standalone functional checks: gradients, hard bounds and reproducibility."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from silence_the_mimic import STM, STMConfig


def encode(waveform):
    return F.normalize(torch.stack([
        waveform.mean(), waveform.square().mean(),
        (waveform[1:] - waveform[:-1]).square().mean(),
        waveform.abs().mean(),
    ]), dim=0)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        time = torch.arange(4096) / 16000
        cls.waveform = 0.2 * torch.sin(2 * torch.pi * 220 * time)
        cls.bank = {
            f"p{index}": F.normalize(torch.tensor([1.0, index + 1.0, 0.1, 0.5]), dim=0)
            for index in range(8)
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_default_80_updates_are_finite_and_feasible(self):
        result = STM(encode).protect(
            self.waveform, sample_rate=16000, referral_bank=self.bank,
        )
        self.assertEqual(result.waveform.shape, self.waveform.shape)
        self.assertTrue(torch.isfinite(result.waveform).all())
        self.assertTrue((result.perturbation >= result.lower_bound).all())
        self.assertTrue((result.perturbation <= result.upper_bound).all())
        self.assertFalse(result.waveform.requires_grad)

    def test_target_rank_and_explicit_target_give_same_waveform(self):
        config = STMConfig(iterations=4)
        original = encode(self.waveform)
        selected = sorted(
            self.bank, key=lambda k: F.mse_loss(self.bank[k], original).item(),
            reverse=True,
        )[5]
        stm = STM(encode, config)
        from_bank = stm.protect(
            self.waveform, sample_rate=16000, referral_bank=self.bank,
        )
        explicit = stm.protect(
            self.waveform, sample_rate=16000, target_embedding=self.bank[selected],
        )
        self.assertEqual(from_bank.target_speaker, selected)
        self.assertIsNone(explicit.target_speaker)
        self.assertTrue(torch.equal(from_bank.waveform, explicit.waveform))

    def test_repeatable_without_changing_input_bank_or_global_rng(self):
        before_rng = torch.get_rng_state().clone()
        before_wave = self.waveform.clone()
        before_bank = {k: v.clone() for k, v in self.bank.items()}
        stm = STM(encode, STMConfig(iterations=4))
        first = stm.protect(self.waveform, sample_rate=16000, referral_bank=self.bank)
        second = stm.protect(self.waveform, sample_rate=16000, referral_bank=self.bank)
        self.assertTrue(torch.equal(first.waveform, second.waveform))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(before_wave, self.waveform))
        self.assertTrue(all(torch.equal(v, self.bank[k]) for k, v in before_bank.items()))

    def test_detached_encoder_is_rejected(self):
        stm = STM(lambda wav: encode(wav).detach(), STMConfig(iterations=1))
        with self.assertRaisesRegex(ValueError, "detached"):
            stm.protect(self.waveform, sample_rate=16000, referral_bank=self.bank)

    def test_encoder_parameter_gradients_and_mode_are_preserved(self):
        model = torch.nn.Linear(4, 4, bias=False)
        model.train()
        stm = STM(lambda wav: model(encode(wav)), STMConfig(iterations=4))
        before = model.weight.detach().clone()
        stm.protect(
            self.waveform, sample_rate=16000, target_embedding=torch.ones(4),
        )
        self.assertIsNone(model.weight.grad)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, model.weight))

    def test_tv_disabled_accepts_short_input_and_ignores_lags(self):
        waveform = self.waveform[:512]
        results = []
        for lags in ((1, 2, 3, 4), (1000,)):
            with self.subTest(lags=lags):
                config = STMConfig(iterations=4, tv_weight=0, tv_lags=lags)
                result = STM(encode, config).protect(
                    waveform, sample_rate=16000, referral_bank=self.bank,
                )
                self.assertEqual(result.waveform.shape, waveform.shape)
                self.assertTrue(torch.isfinite(result.waveform).all())
                self.assertTrue((result.perturbation >= result.lower_bound).all())
                self.assertTrue((result.perturbation <= result.upper_bound).all())
                results.append(result)
        self.assertTrue(torch.equal(results[0].waveform, results[1].waveform))
        self.assertTrue(torch.equal(results[0].perturbation, results[1].perturbation))

    def test_tv_disabled_does_not_compute_tv_reductions(self):
        stm = STM(encode, STMConfig(iterations=4, tv_weight=0))
        # TV uses torch.mean; the encoder and speaker loss use Tensor.mean.
        with patch("silence_the_mimic.core.torch.mean", wraps=torch.mean) as tv_mean:
            stm.protect(self.waveform, sample_rate=16000, referral_bank=self.bank)
        tv_mean.assert_not_called()

    def test_tv_enabled_still_rejects_short_input(self):
        stm = STM(encode, STMConfig(iterations=1))
        with self.assertRaisesRegex(ValueError, "configured TV lags"):
            stm.protect(self.waveform[:512], sample_rate=16000, referral_bank=self.bank)

    def test_tv_disabled_keeps_stft_length_check(self):
        stm = STM(encode, STMConfig(iterations=1, tv_weight=0))
        with self.assertRaisesRegex(ValueError, "STFT reflection padding"):
            stm.protect(self.waveform[:256], sample_rate=16000, referral_bank=self.bank)

    def test_invalid_inputs_fail_explicitly(self):
        stm = STM(encode, STMConfig(iterations=1))
        for invalid in (
            self.waveform.unsqueeze(0), self.waveform.double(),
            self.waveform[:100], torch.full((4096,), float("nan")),
            torch.full((4096,), 1.1),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    stm.protect(invalid, sample_rate=16000, referral_bank=self.bank)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            stm.protect(self.waveform, sample_rate=16000)
        with self.assertRaisesRegex(ValueError, "requires"):
            stm.protect(self.waveform, sample_rate=16000, referral_bank={"p0": torch.ones(4)})


if __name__ == "__main__":
    unittest.main()
