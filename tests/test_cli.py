"""Runner checks with generated audio; no external models or downloads."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.minimal import toy_encoder
from silence_the_mimic.cli import check_bank_receipt, load_waveform, main, plan_outputs, save_result
from silence_the_mimic.encoders import create_attack_encoder, sha256_file


class EncoderModeTests(unittest.TestCase):
    def test_original_attack_modes_are_frozen(self):
        for name in ("freevc", "quickvc", "triaanvc", "gpt_sovits"):
            with self.subTest(model=name):
                model = torch.nn.Sequential(torch.nn.Linear(1, 1))
                model.provenance = {"model": name}
                with patch("silence_the_mimic.encoders.create_encoder", return_value=model):
                    encoder = create_attack_encoder(name, model_root="unused", checkpoint="unused")
                self.assertEqual(encoder.sample_rate, 16000)
                self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
                self.assertTrue(all(module.training == (name != "gpt_sovits") for module in encoder.modules()))
                self.assertEqual(encoder.provenance["mode"], "eval" if name == "gpt_sovits" else "train")

    @unittest.skipUnless(importlib.util.find_spec("torchaudio"), 'Install ".[encoders]"')
    def test_gsv_resampling_preserves_input_gradient(self):
        class Native32k(torch.nn.Module):
            def forward(self, wav):
                self.length = len(wav)
                return wav.square().mean().unsqueeze(0)
        model = Native32k()
        model.provenance = {"model": "gpt_sovits"}
        with patch("silence_the_mimic.encoders.create_encoder", return_value=model):
            encoder = create_attack_encoder("gpt_sovits", model_root="unused", checkpoint="unused")
        wav = torch.linspace(-0.2, 0.2, 4096, requires_grad=True)
        gradient, = torch.autograd.grad(encoder(wav).sum(), wav)
        self.assertEqual(model.length, 8192)
        self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().max() > 0)


@unittest.skipUnless(importlib.util.find_spec("soundfile"), 'Install ".[audio]"')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        import soundfile as sf

        temporary = tempfile.TemporaryDirectory(prefix="stm-cli-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.source = self.root / "clean.wav"
        self.waveform = (0.2 * np.sin(2 * np.pi * 220 * np.arange(4096) / 16000)).astype(np.float32)
        sf.write(self.source, self.waveform, 16000, subtype="FLOAT")
        self.bank = self.root / "bank.pt"
        torch.save({str(i): torch.tensor([1., i + 1., .1, .5]) for i in range(8)}, self.bank)
        self.output = self.root / "out" / "protected.wav"

    def command(self, source=None, output=None):
        return ["--model", "freevc", "--model-root", str(self.root), "--checkpoint", "unused",
                "--bank", str(self.bank), "--input", str(source or self.source),
                "--output", str(output or self.output), "--iterations", "4", "--seed", "42"]

    def run_cli(self, command):
        def encode(wav):
            return toy_encoder(wav)
        encode.provenance = {"model": "freevc", "mode": "train", "source_sha256": {}}
        with patch("silence_the_mimic.cli.create_attack_encoder", return_value=encode), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(command)

    def test_load_does_not_change_float_audio(self):
        self.assertTrue(np.array_equal(load_waveform(self.source), self.waveform))

    def test_reject_wrong_rate_stereo_nonfinite_and_overshoot(self):
        import soundfile as sf

        for waveform, rate in ((self.waveform, 32000), (np.stack([self.waveform] * 2, axis=1), 16000),
                               (np.array([np.nan], np.float32), 16000), (np.array([1.2], np.float32), 16000)):
            with self.subTest(rate=rate, shape=waveform.shape):
                invalid = self.root / "invalid.wav"
                sf.write(invalid, waveform, rate, subtype="FLOAT")
                with self.assertRaises(ValueError):
                    load_waveform(invalid)

    def test_float_export_preserves_overshoot(self):
        raw = torch.tensor([1.03, -1.02, 0.0])
        save_result(SimpleNamespace(waveform=raw), self.output, {})
        import soundfile as sf

        actual, rate = sf.read(self.output, dtype="float32")
        self.assertTrue(np.array_equal(actual, raw.numpy()))
        self.assertEqual(rate, 16000)
        receipt = json.loads(self.output.with_suffix(".wav.json").read_text())
        self.assertFalse(receipt["output_clipping"])
        self.assertEqual(receipt["output_sha256"], sha256_file(self.output))

    def test_pcm16_rejects_overshoot_before_writing(self):
        with self.assertRaisesRegex(ValueError, "clip"):
            save_result(SimpleNamespace(waveform=torch.tensor([1.1])), self.output, {}, "PCM_16")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".wav.json").exists())

    def test_no_overwrite_even_if_output_appears_after_preflight(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            save_result(SimpleNamespace(waveform=torch.zeros(10)), self.output, {})
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertFalse(self.output.with_suffix(".wav.json").exists())

    def test_failed_write_cleans_only_new_partial_files(self):
        with patch("soundfile.write", side_effect=RuntimeError("write failed")), self.assertRaises(RuntimeError):
            save_result(SimpleNamespace(waveform=torch.zeros(10)), self.output, {})
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".wav.json").exists())

    def test_directory_planning_and_full_preflight(self):
        input_dir = self.root / "batch"
        (input_dir / "nested").mkdir(parents=True)
        (input_dir / "b.wav").write_bytes(self.source.read_bytes())
        (input_dir / "nested" / "a.wav").write_bytes(self.source.read_bytes())
        output_dir = self.root / "batch-output"
        pairs = plan_outputs(input_dir, output_dir)
        self.assertEqual([path.relative_to(input_dir).as_posix() for path, _ in pairs], ["b.wav", "nested/a.wav"])
        with self.assertRaises(ValueError):
            plan_outputs(input_dir, input_dir / "protected")
        pairs[-1][1].parent.mkdir(parents=True)
        pairs[-1][1].with_suffix(".wav.json").write_text("existing")
        with self.assertRaises(FileExistsError):
            plan_outputs(input_dir, output_dir)
        self.assertFalse(pairs[0][1].exists())

    def test_cli_full_protection_and_receipt(self):
        command = self.command()
        self.run_cli(command)
        metadata = json.loads(self.output.with_suffix(".wav.json").read_text())
        self.assertEqual(metadata["config"]["iterations"], 4)
        self.assertEqual(metadata["seed"], 42)
        self.assertIsNone(metadata["trim_top_db"])
        self.assertFalse(metadata["bank_encoder_metadata_checked"])
        self.assertEqual(metadata["output_subtype"], "FLOAT")
        self.assertTrue(torch.isfinite(torch.from_numpy(load_waveform(self.output))).all())
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_cli(command)
        self.assertEqual(self.output.read_bytes(), before)

    def test_batch_seeds_and_nested_output(self):
        batch = self.root / "batch"
        (batch / "nested").mkdir(parents=True)
        (batch / "a.wav").write_bytes(self.source.read_bytes())
        (batch / "nested" / "b.wav").write_bytes(self.source.read_bytes())
        output = self.root / "batch-output"
        self.run_cli(self.command(batch, output))
        for index, filename in enumerate(("a.wav.json", "nested/b.wav.json")):
            metadata = json.loads((output / filename).read_text())
            self.assertEqual(metadata["seed"], 42 + index)

    def test_failed_input_is_not_silently_skipped(self):
        import soundfile as sf

        sf.write(self.source, self.waveform, 8000)
        with self.assertRaisesRegex(RuntimeError, "Failed input.*stopped without skipping"):
            self.run_cli(self.command())
        self.assertFalse(self.output.exists())

    def test_bank_receipt_checks_model_rate_and_checkpoint(self):
        checkpoint = self.root / "encoder.pt"
        checkpoint.write_bytes(b"checkpoint")
        metadata = {"model": {"model": "freevc", "checkpoint_sha256": sha256_file(checkpoint)},
                    "preprocessing": {"sample_rate": 16000}}
        receipt = self.bank.with_suffix(".pt.json")
        receipt.write_text(json.dumps(metadata))
        self.assertEqual(check_bank_receipt(self.bank, "freevc", checkpoint), metadata)
        for key, value in (("model", "quickvc"), ("checkpoint_sha256", "wrong")):
            bad = json.loads(json.dumps(metadata))
            bad["model"][key] = value
            receipt.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                check_bank_receipt(self.bank, "freevc", checkpoint)
        metadata["preprocessing"]["sample_rate"] = 8000
        receipt.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "sample rate"):
            check_bank_receipt(self.bank, "freevc", checkpoint)

    def test_triaan_bank_receipt_checks_cpc_checkpoint(self):
        checkpoint, cpc = self.root / "encoder.pt", self.root / "cpc.pt"
        checkpoint.write_bytes(b"encoder")
        cpc.write_bytes(b"cpc")
        metadata = {"model": {"model": "triaanvc", "checkpoint_sha256": sha256_file(checkpoint),
                              "cpc_sha256": sha256_file(cpc)}, "preprocessing": {"sample_rate": 16000}}
        receipt = self.bank.with_suffix(".pt.json")
        receipt.write_text(json.dumps(metadata))
        self.assertEqual(check_bank_receipt(self.bank, "triaanvc", checkpoint, cpc), metadata)
        cpc.write_bytes(b"different cpc")
        with self.assertRaisesRegex(ValueError, "CPC"):
            check_bank_receipt(self.bank, "triaanvc", checkpoint, cpc)

    def test_mismatched_bank_stops_before_model_loading_or_writing(self):
        receipt = self.bank.with_suffix(".pt.json")
        receipt.write_text(json.dumps({"model": {"model": "gpt_sovits"}}))
        with patch("silence_the_mimic.cli.create_attack_encoder") as factory, self.assertRaisesRegex(ValueError, "different model"):
            main(self.command())
        factory.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
