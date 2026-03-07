import unittest
import sys
import os
import torch
import torch.nn as nn

# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.models.factory as mf
import onconet.models.default_models
import onconet.models  # noqa: F401 - triggers all @RegisterModel decorators


class Args():
    pass


class TestModels(unittest.TestCase):
    def setUp(self):
        self.args = Args()
        self.args.weight_decay = 5e-5
        self.args.lr = 0.001
        self.model = nn.Linear(10, 2)

    def tearDown(self):
        self.args = None
        self.model = None

    def test_get_existing_optimizers(self):
        args = self.args
        optimizers = [
            ('adam', torch.optim.Adam),
        ]
        for optimizer, optim_type in optimizers:
            args.optimizer = optimizer
            optim = mf.get_optimizer(self.model, args)
            self.assertIsInstance(optim, optim_type)

    def test_non_existing_optimizers(self):
        args = self.args
        optimizers = [
            None,
            'yala',
            5,
        ]
        for optimizer in optimizers:
            args.optimizer = optimizer
            with self.assertRaises(Exception) as context:
                mf.get_optimizer(self.model, args)

            self.assertTrue(
                'Optimizer {} not supported!'.format(optimizer) in str(
                    context.exception))


class TestMiraiFull25dAttnRegistration(unittest.TestCase):
    """Smoke-tests for the mirai_full_25d_attn model."""

    def test_model_is_registered(self):
        """mirai_full_25d_attn must appear in MODEL_REGISTRY after importing onconet.models."""
        self.assertIn('mirai_full_25d_attn', mf.MODEL_REGISTRY)

    def _build_args(self, **overrides):
        """Build a minimal args namespace sufficient to construct the model."""
        args = Args()
        # ResNet / encoder defaults
        args.block_layout = [
            "BasicBlock,1",
            "BasicBlock,1",
            "BasicBlock,1",
            "BasicBlock,1",
        ]
        args.block_widening_factor = 1
        args.num_groups = 1
        args.pool_name = 'GlobalAvgPool'
        args.use_risk_factors = False
        args.deep_risk_factor_pool = False
        args.use_precomputed_hiddens = False
        args.use_spatial_transformer = False
        args.predict_birads = False
        args.survival_analysis_setup = False
        args.use_region_annotation = False
        args.num_classes = 2
        args.num_chan = 1
        args.input_dim = 512
        args.dropout = 0.0
        args.cuda = False
        args.model_parallel = False
        args.num_shards = 1
        args.img_only_dim = 64
        args.pretrained_on_imagenet = False
        args.pretrained_imagenet_model_name = 'resnet18'
        # mirai_full_25d_attn defaults
        args.img_encoder_snapshot = None
        args.freeze_image_encoder = False
        args.slice_encoder_chunk_size = 4
        args.slice_attn_dropout = 0.0
        args.slice_token_drop = 0.0
        # factory / wrap
        args.model_name = 'mirai_full_25d_attn'
        args.wrap_model = False
        args.state_dict_path = None
        args.num_gpus = 1
        args.data_parallel = False
        args.multi_image = False
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_forward_shape(self):
        """Model should produce (logit, hidden, activ_dict) with correct shapes."""
        args = self._build_args()
        model = mf.get_model_by_name('mirai_full_25d_attn', False, args)
        model.eval()

        B, D, C, H, W = 2, 8, 1, 32, 32
        x = torch.zeros(B, D, C, H, W)
        logit, hidden, activ_dict = model(x)

        self.assertEqual(logit.shape, (B, args.num_classes))
        self.assertEqual(hidden.ndim, 2)
        self.assertEqual(hidden.shape[0], B)
        self.assertIn('activ', activ_dict)
        self.assertIn('attn_weights', activ_dict)
        # attn_weights must sum to ~1 over depth
        attn = activ_dict['attn_weights']
        self.assertEqual(attn.shape, (B, D))
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(B), atol=1e-5))

    def test_freeze_image_encoder(self):
        """With freeze_image_encoder=True, no encoder param should require grad."""
        args = self._build_args(freeze_image_encoder=True)
        model = mf.get_model_by_name('mirai_full_25d_attn', False, args)
        for name, p in model.image_encoder.named_parameters():
            self.assertFalse(
                p.requires_grad,
                msg=f"Parameter {name} should be frozen but requires_grad=True",
            )

    def test_chunked_encoding_matches(self):
        """Different chunk sizes should produce the same slice features."""
        args1 = self._build_args(slice_encoder_chunk_size=2)
        args2 = self._build_args(slice_encoder_chunk_size=4)
        model1 = mf.get_model_by_name('mirai_full_25d_attn', False, args1)
        model2 = mf.get_model_by_name('mirai_full_25d_attn', False, args2)
        # Share weights so the only difference is chunk size
        model2.load_state_dict(model1.state_dict())
        model1.eval()
        model2.eval()

        x = torch.randn(1, 4, 1, 32, 32)
        with torch.no_grad():
            logit1, _, _ = model1(x)
            logit2, _, _ = model2(x)
        self.assertTrue(torch.allclose(logit1, logit2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
