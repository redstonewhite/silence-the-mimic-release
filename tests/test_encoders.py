"""Checkpoint validation without upstream model assets or audio dependencies."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from silence_the_mimic.encoders import create_encoder


class LegacyLenientEncoder(torch.nn.Module):
    """Emulate the upstream FreeVC constructor's non-strict loading."""

    def __init__(self, checkpoint, device, verbose):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(state["model_state"], strict=False)
        self.to(device)


class FreeVCCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stm-checkpoint-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "speaker_encoder").mkdir()
        for name in ("voice_encoder.py", "hparams.py"):
            (self.root / "speaker_encoder" / name).write_text("# test fixture\n")
        self.checkpoint = self.root / "encoder.pt"
        self.valid = {"linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                      "linear.bias": torch.tensor([0.25, -0.5])}
        self.constructor = Mock(side_effect=LegacyLenientEncoder)
        importer = patch("silence_the_mimic.encoders._import_model",
                         return_value=SimpleNamespace(SpeakerEncoder=self.constructor))
        importer.start()
        self.addCleanup(importer.stop)
        wrapper = patch("silence_the_mimic.encoders._FreeVC", side_effect=lambda model: torch.nn.Sequential(model))
        wrapper.start()
        self.addCleanup(wrapper.stop)

    def load(self, state):
        torch.save(state, self.checkpoint)
        return create_encoder("freevc", model_root=self.root, checkpoint=self.checkpoint)

    def test_complete_checkpoint_loads_exact_weights_and_freezes_them(self):
        encoder = self.load({"model_state": self.valid, "step": 123})
        for name, expected in self.valid.items():
            self.assertTrue(torch.equal(encoder[0].state_dict()[name], expected))
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_missing_empty_or_invalid_model_state_fails_before_construction(self):
        for state in ({}, {"model_state": {}}, {"model_state": None}, {"model_state": []}, []):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "nonempty model_state"):
                self.load(state)
        self.constructor.assert_not_called()

    def test_partial_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load({"model_state": {"linear.weight": self.valid["linear.weight"]}})

    def test_known_training_loss_scalars_do_not_change_inference_weights(self):
        state = dict(self.valid, similarity_weight=torch.tensor([10.0]), similarity_bias=torch.tensor([-5.0]))
        encoder = self.load({"model_state": state})
        for name, expected in self.valid.items():
            self.assertTrue(torch.equal(encoder[0].state_dict()[name], expected))

    def test_training_loss_scalars_cannot_replace_missing_inference_parameters(self):
        state = {"similarity_weight": torch.tensor([10.0]), "similarity_bias": torch.tensor([-5.0])}
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load({"model_state": state})

    def test_unexpected_parameter_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.load({"model_state": dict(self.valid, unexpected=torch.ones(1))})

    def test_wrong_parameter_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.load({"model_state": dict(self.valid, **{"linear.weight": torch.ones(3, 3)})})


if __name__ == "__main__":
    unittest.main()
