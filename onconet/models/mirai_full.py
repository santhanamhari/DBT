from onconet.models.factory import load_model, RegisterModel, get_model_by_name
import torch
import torch.nn as nn


@RegisterModel("mirai_full")
class MiraiFull(nn.Module):
    """2.5D Mirai model: per-slice 2D encoding + stats pooling + view transformer.

    Accepts input shaped (B, Nviews, D, C, H, W).

    Pipeline:
    1. Shared 2D image encoder processes all slices (chunked over D to avoid OOM).
    2. Order-invariant depth aggregation via mean/std/min/max statistics.
    3. Small fusion MLP projects (B, N, 4F) → (B, N, F).
    4. Existing MIRAI transformer aggregates across views.

    Backward-compatible external interface:
        forward(x, risk_factors=None, batch=None) → (logit, hidden, activ_dict)
    """

    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args

        # --- image encoder (same pretrained weights as original mirai_full) ---
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim

        # --- view transformer (same as original mirai_full) ---
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name('transformer', False, args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

        # --- chunking over depth dimension to avoid OOM ---
        self.slice_encoder_chunk_size = getattr(args, 'slice_encoder_chunk_size', 2)

        # --- depth stats fusion: 4F → F ---
        p = getattr(args, 'depth_stats_dropout', 0.2)
        self.depth_stats_fuse = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(4 * self.image_repr_dim, self.image_repr_dim),
            nn.ReLU(inplace=True),
        )

    def _encode_view_slices(self, x_bn: torch.Tensor) -> torch.Tensor:
        """Encode all slices for a batch of views via the shared 2D image encoder.

        Args:
            x_bn: (B*N, D, C, H, W) — flattened batch × view dimension.

        Returns:
            Tensor of shape (B*N, D, F) containing per-slice feature vectors.
        """
        BN, D, C, H, W = x_bn.shape
        chunk = self.slice_encoder_chunk_size
        freeze = hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder

        feats = []
        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            n = end - start
            x_chunk = x_bn[:, start:end].contiguous().view(BN * n, C, H, W)

            if freeze:
                with torch.no_grad():
                    _, h, _ = self.image_encoder(x_chunk)
            else:
                _, h, _ = self.image_encoder(x_chunk)

            h = h.view(BN, n, -1)[:, :, :self.image_repr_dim]  # (BN, n, F)
            feats.append(h)

        return torch.cat(feats, dim=1)  # (BN, D, F)

    def forward(self, x, risk_factors=None, batch=None):
        """Forward pass.

        Args:
            x: (B, Nviews, D, C, H, W)
            risk_factors: optional list of risk factor tensors
            batch: optional batch metadata dict

        Returns:
            logit, hidden, activ_dict
        """
        B, N, D, C, H, W = x.size()

        # Flatten views into batch dimension for slice encoding
        x_bn = x.view(B * N, D, C, H, W)

        # Per-slice encoding (chunked over D)
        slice_feats = self._encode_view_slices(x_bn)              # (B*N, D, F)
        slice_feats = slice_feats.view(B, N, D, self.image_repr_dim)  # (B, N, D, F)

        # Order-invariant depth aggregation via distribution statistics.
        # unbiased=False (biased std) is used for numerical stability when D is small;
        # the biased estimator is well-defined even for D=1, unlike the unbiased one.
        mu = slice_feats.mean(dim=2)                               # (B, N, F)
        sd = slice_feats.std(dim=2, unbiased=False)                # (B, N, F)
        mn = slice_feats.min(dim=2).values                         # (B, N, F)
        mx = slice_feats.max(dim=2).values                         # (B, N, F)

        # Fuse stats into a single per-view feature vector
        view_feats = self.depth_stats_fuse(
            torch.cat([mu, sd, mn, mx], dim=-1)
        )                                                           # (B, N, F)

        # View aggregation via MIRAI transformer
        logit, transformer_hidden, activ_dict = self.transformer(view_feats, risk_factors, batch)

        activ_dict = dict(activ_dict) if activ_dict is not None else {}
        activ_dict['slice_feats'] = slice_feats
        activ_dict['depth_mu'] = mu
        activ_dict['depth_sd'] = sd
        activ_dict['depth_min'] = mn
        activ_dict['depth_max'] = mx

        return logit, transformer_hidden, activ_dict
