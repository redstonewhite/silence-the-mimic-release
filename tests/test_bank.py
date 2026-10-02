"""Bank construction checks using generated PCM audio and a toy encoder."""

import csv
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch

from examples.build_bank import build_bank, main, read_manifest, vctk_records
from examples.minimal import toy_encoder
from silence_the_mimic import STM, STMConfig


@unittest.skipUnless(importlib.util.find_spec("librosa") is not None, 'Install ".[audio]" for bank tests')
class BankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stm-bank-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.original_threads)
        self.manifest = self.root / "samples.csv"
        self.speakers = ["p340", "p225", "p268", "p226", "p295", "p227"]
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("speaker", "utterance", "wav_path"))
            writer.writeheader()
            for index, speaker in enumerate(self.speakers):
                folder = self.root / speaker
                folder.mkdir()
                number = "021" if speaker in {"p268", "p295", "p340"} else "023"
                utterance = f"{speaker}_{number}"
                path = folder / f"{utterance}.wav"
                samples = (6000 * np.sin(2 * np.pi * (220 + index * 30) * np.arange(4096) / 16000)).astype("<i2")
                with wave.open(str(path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(samples.tobytes())
                writer.writerow({"speaker": speaker, "utterance": utterance,
                                 "wav_path": str(path.relative_to(self.root))})

    def test_manifest_preserves_order_and_resolves_relative_paths(self):
        records = read_manifest(self.manifest)
        self.assertEqual([row["speaker"] for row in records], self.speakers)
        self.assertTrue(all(Path(row["wav_path"]).is_absolute() for row in records))

    def test_vctk_fixed_selection_and_exceptions(self):
        records = vctk_records(self.root)
        self.assertEqual([row["speaker"] for row in records], sorted(self.speakers))
        for row in records:
            expected = "021" if row["speaker"] in {"p268", "p295", "p340"} else "023"
            self.assertEqual(row["utterance"], f"{row['speaker']}_{expected}")

    def test_original_filesystem_order_is_preserved_when_requested(self):
        expected = [path.name for path in self.root.iterdir() if path.is_dir()]
        self.assertEqual([row["speaker"] for row in vctk_records(self.root, sort=False)], expected)

    def test_original_gsv_sample_rate_cannot_silently_change(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--vctk-root", str(self.root), "--model", "gpt_sovits",
                  "--model-root", str(self.root), "--checkpoint", "unused.pth",
                  "--sample-rate", "16000", "--output", str(self.root / "invalid.pt")])
        self.assertFalse((self.root / "invalid.pt").exists())

    def test_original_model_requires_explicit_assets(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--vctk-root", str(self.root), "--model", "freevc",
                  "--output", str(self.root / "invalid.pt")])
        self.assertFalse((self.root / "invalid.pt").exists())

    def test_triaan_compatibility_mode_cannot_change_other_models(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--vctk-root", str(self.root), "--model", "freevc",
                  "--model-root", str(self.root), "--checkpoint", "unused.pth",
                  "--triaan-bank-mode", "train", "--output", str(self.root / "invalid.pt")])
        self.assertFalse((self.root / "invalid.pt").exists())

    def test_missing_sample_fails_instead_of_falling_back(self):
        with self.assertRaises(FileNotFoundError):
            vctk_records(self.root, "_missing.wav")

    def test_duplicate_speakers_are_rejected(self):
        with self.manifest.open("a", newline="") as handle:
            handle.write("p225,p225_023,p225/p225_023.wav\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_manifest(self.manifest)

    def test_bank_matches_direct_encoding_without_extra_normalization(self):
        import librosa

        records = read_manifest(self.manifest)
        encode = lambda wav: torch.stack([wav.mean(), wav.square().mean(), wav.abs().mean()])
        bank = build_bank(encode, records)
        self.assertEqual(list(bank), self.speakers)
        for row in records:
            waveform, _ = librosa.load(row["wav_path"], sr=16000, mono=True, dtype=np.float32, res_type="soxr_hq")
            expected = encode(torch.from_numpy(waveform))
            self.assertTrue(torch.equal(bank[row["speaker"]], expected))
            self.assertFalse(bank[row["speaker"]].requires_grad)
            self.assertEqual(bank[row["speaker"]].device.type, "cpu")

    def test_invalid_embedding_reports_sample_and_fails(self):
        with self.assertRaisesRegex(ValueError, "Failed bank sample p340/p340_021"):
            build_bank(lambda wav: torch.tensor([float("nan")]), read_manifest(self.manifest))

    def test_resampling_overshoot_is_not_clipped_or_normalized(self):
        waveform = np.array([1.04, -1.02, 0.0], dtype=np.float32)
        with patch("librosa.load", return_value=(waveform, 16000)):
            bank = build_bank(lambda wav: torch.stack([wav.max(), wav.min()]), read_manifest(self.manifest))
        expected = torch.tensor([1.04, -1.02])
        self.assertTrue(all(torch.equal(value, expected) for value in bank.values()))

    def test_cli_bank_loads_in_stm_and_refuses_overwrite(self):
        output = self.root / "artifacts" / "bank.pt"
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        main(command)
        bank = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(list(bank), self.speakers)
        receipt = output.with_suffix(".pt.json")
        metadata = json.loads(receipt.read_text())
        self.assertEqual(metadata["selection"], "explicit_manifest")
        self.assertFalse(metadata["preprocessing"]["trim"])
        self.assertEqual(metadata["samples"], read_manifest(self.manifest))
        waveform = 0.2 * torch.sin(2 * torch.pi * 220 * torch.arange(4096) / 16000)
        result = STM(toy_encoder, STMConfig(iterations=4)).protect(
            waveform, sample_rate=16000, referral_bank=bank,
        )
        self.assertTrue(torch.isfinite(result.waveform).all())
        before = output.read_bytes(), receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            main(command)
        self.assertEqual(before, (output.read_bytes(), receipt.read_bytes()))

    def test_receipt_write_failure_removes_partials_and_allows_retry(self):
        output = self.root / "artifacts" / "bank.pt"
        receipt = output.with_suffix(".pt.json")
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        def fail_after_partial_write(metadata, handle, **kwargs):
            handle.write("{\n")
            raise OSError("receipt write failed")
        with patch("examples.build_bank.json.dump", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "receipt write failed"):
                main(command)
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())
        main(command)
        self.assertEqual(len(torch.load(output, map_location="cpu", weights_only=True)), 6)
        self.assertEqual(json.loads(receipt.read_text())["samples"], read_manifest(self.manifest))

    def test_bank_write_failure_removes_partials_and_allows_retry(self):
        output = self.root / "artifacts" / "bank.pt"
        receipt = output.with_suffix(".pt.json")
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        def fail_after_partial_write(bank, handle):
            handle.write(b"partial bank")
            raise OSError("bank write failed")
        with patch("examples.build_bank.torch.save", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "bank write failed"):
                main(command)
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())
        main(command)
        self.assertTrue(output.is_file() and receipt.is_file())

    def test_keyboard_interrupt_removes_created_partials(self):
        output = self.root / "bank.pt"
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        with patch("examples.build_bank.torch.save", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                main(command)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".pt.json").exists())

    def test_bank_appearing_after_preflight_is_preserved(self):
        output = self.root / "bank.pt"
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        def concurrent_output(*args, **kwargs):
            bank = build_bank(*args, **kwargs)
            output.write_bytes(b"other job's bank")
            return bank
        with patch("examples.build_bank.build_bank", side_effect=concurrent_output):
            with self.assertRaises(FileExistsError):
                main(command)
        self.assertEqual(output.read_bytes(), b"other job's bank")
        self.assertFalse(output.with_suffix(".pt.json").exists())

    def test_receipt_appearing_after_preflight_is_preserved(self):
        output = self.root / "bank.pt"
        receipt = output.with_suffix(".pt.json")
        command = ["--manifest", str(self.manifest), "--encoder", "examples.minimal:create_encoder",
                   "--output", str(output)]
        def concurrent_receipt(*args, **kwargs):
            bank = build_bank(*args, **kwargs)
            receipt.write_text("other job's receipt")
            return bank
        with patch("examples.build_bank.build_bank", side_effect=concurrent_receipt):
            with self.assertRaises(FileExistsError):
                main(command)
        self.assertEqual(receipt.read_text(), "other job's receipt")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
