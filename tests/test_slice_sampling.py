"""Tests for grouped slice sampling and MiraiFull 2.5D forward shape."""
import sys
import os
import types
import importlib.util
import unittest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so the loader module can be imported
# without lifelines, pydicom, etc.
# ---------------------------------------------------------------------------

def _stub_module(name):
    """Insert a stub module into sys.modules to prevent import errors."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod


# Stub the dicom_multiframe sub-module before importing image.py
_stub_module("onconet")
_stub_module("onconet.datasets")
_stub_module("onconet.datasets.loader")
_loader_stub = types.ModuleType("onconet.datasets.loader.dicom_multiframe")
_loader_stub.load_multiframe_dicom = None  # not used in these tests
_loader_stub.normalize_minmax = None
sys.modules["onconet.datasets.loader.dicom_multiframe"] = _loader_stub

# Insert the repo root so the loader file can be found
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import the loader module directly to avoid triggering onconet/datasets/__init__.py
# which pulls in many optional dependencies.
_loader_path = os.path.join(_REPO_ROOT, "onconet", "datasets", "loader", "image.py")
_spec = importlib.util.spec_from_file_location("onconet.datasets.loader.image", _loader_path)
image_loader_module = importlib.util.module_from_spec(_spec)
sys.modules["onconet.datasets.loader.image"] = image_loader_module
_spec.loader.exec_module(image_loader_module)  # type: ignore[attr-defined]


class TestGroupedSliceSampling(unittest.TestCase):
    """Tests for _select_slice_indices with the grouped policy."""

    def setUp(self):
        self.loader = image_loader_module.image_loader(None, [])

    # ------------------------------------------------------------------
    # grouped (7×3 = 21 slices) from a 70-frame volume
    # ------------------------------------------------------------------
    def test_grouped_returns_21_indices(self):
        idxs = self.loader._select_slice_indices(70, 21, "grouped")
        self.assertEqual(len(idxs), 21)

    def test_grouped_indices_in_range(self):
        idxs = self.loader._select_slice_indices(70, 21, "grouped")
        self.assertTrue(all(0 <= i < 70 for i in idxs), f"Out-of-range index in {idxs}")

    def test_grouped_7x3_alias_matches_grouped(self):
        """grouped_7x3 should produce the same result as grouped (no jitter)."""
        # Use a fixed seed to make jitter deterministic; with jitter=0 both should match.
        idxs_grouped = self.loader._select_slice_indices(70, 21, "grouped", jitter=0)
        idxs_alias = self.loader._select_slice_indices(70, 21, "grouped_7x3", jitter=0)
        self.assertEqual(idxs_grouped, idxs_alias)
    def test_grouped_zero_target_returns_all(self):
        """num_slices=0 should return all slice indices."""
        idxs = self.loader._select_slice_indices(70, 0, "grouped")
        self.assertEqual(idxs, list(range(70)))

    def test_grouped_none_target_returns_all(self):
        """num_slices=None should return all slice indices."""
        idxs = self.loader._select_slice_indices(70, None, "grouped")
        self.assertEqual(idxs, list(range(70)))

    def test_grouped_jitter_within_bounds(self):
        """With jitter, indices must still be in [0, num_frames)."""
        for _ in range(20):
            idxs = self.loader._select_slice_indices(70, 21, "grouped", jitter=5)
            self.assertTrue(all(0 <= i < 70 for i in idxs), f"Out-of-range with jitter: {idxs}")

    def test_grouped_covers_full_depth(self):
        """Bin centers should span the full volume depth (first bin near start, last near end)."""
        idxs = self.loader._select_slice_indices(70, 21, "grouped", jitter=0)
        # The first bin center is around frame 5 (0.5/7 * 70 = 5)
        # The last bin center is around frame 65 (6.5/7 * 70 ≈ 65)
        self.assertLessEqual(min(idxs), 10, "Grouped sampling should cover near the start")
        self.assertGreaterEqual(max(idxs), 60, "Grouped sampling should cover near the end")

    # ------------------------------------------------------------------
    # Legacy policies still work
    # ------------------------------------------------------------------
    def test_center_crop_returns_correct_count(self):
        idxs = self.loader._select_slice_indices(70, 21, "center_crop")
        self.assertEqual(len(idxs), 21)

    def test_uniform_returns_correct_count(self):
        idxs = self.loader._select_slice_indices(70, 21, "uniform")
        self.assertEqual(len(idxs), 21)

    # ------------------------------------------------------------------
    # Edge: volume smaller than target_slices
    # ------------------------------------------------------------------
    def test_grouped_small_volume_fallback(self):
        """When num_frames < target_slices the grouped policy pads via bin clamping.

        The short-circuit (return all frames) only fires when num_frames == target_slices.
        With num_frames=10 and target_slices=21, _grouped_slice_indices runs and returns
        21 indices, all clamped to [0, num_frames-1].
        """
        idxs = self.loader._select_slice_indices(10, 21, "grouped")
        self.assertEqual(len(idxs), 21)
        self.assertTrue(all(0 <= i < 10 for i in idxs))


class TestMiraiFull25dForwardShape(unittest.TestCase):
    """Lightweight integration test for MiraiFull 2.5D forward pass shape.

    This test instantiates MiraiFull with a tiny stub image encoder and
    transformer so that no real pretrained weights are needed.
    """

    def _build_args(self):
        import argparse
        args = argparse.Namespace(
            img_encoder_snapshot=None,
            transformer_snapshot=None,
            freeze_image_encoder=False,
            slice_encoder_chunk_size=2,
            depth_stats_dropout=0.0,
            # custom_resnet params expected by get_model_by_name
            model_name="mirai_full",
            wrap_model=False,
            data_parallel=False,
            num_gpus=1,
            distributed=False,
            factory_wrap_ddp=False,
            # needed by custom_resnet
            block_layout=[
                [("BasicBlock", 1)],
                [("BasicBlock", 1)],
                [("BasicBlock", 1)],
                [("BasicBlock", 1)],
            ],
            block_widening_factor=1,
            num_groups=1,
            pool_name="GlobalAvgPool",
            pretrained_on_imagenet=False,
            pretrained_imagenet_model_name="resnet18",
            replace_bn_with_gn=False,
            num_chan=1,
            img_size=[32, 32],
            dropout=0.0,
            num_classes=2,
            make_fc=False,
            hidden_dim=256,
            img_only_dim=256,
            use_precomputed_hiddens=False,
            use_pred_risk_factors_at_test=False,
            use_pred_risk_factors_if_unk=False,
            pred_risk_factors=False,
            use_spatial_transformer=False,
            state_dict_path=None,
            # transformer
            precomputed_hidden_dim=256,
            transfomer_hidden_dim=256,
            num_heads=4,
            num_layers=2,
        )
        return args

    @unittest.skip("Requires full model dependencies; run manually when snapshots are available")
    def test_forward_output_shape(self):
        """MiraiFull should return (B, num_classes) logits for (B, N, D, C, H, W) input."""
        import torch
        from onconet.models import mirai_full  # noqa: F401 — registers the model
        from onconet.models.factory import get_model_by_name

        args = self._build_args()
        model = get_model_by_name("mirai_full", False, args)
        model.eval()

        B, N, D, C, H, W = 1, 2, 4, 1, 32, 32
        x = torch.zeros(B, N, D, C, H, W)
        with torch.no_grad():
            logit, hidden, activ_dict = model(x)

        self.assertEqual(logit.shape[0], B)
        self.assertIn("depth_mu", activ_dict)
        self.assertIn("depth_sd", activ_dict)
        self.assertIn("depth_min", activ_dict)
        self.assertIn("depth_max", activ_dict)


if __name__ == "__main__":
    unittest.main()
