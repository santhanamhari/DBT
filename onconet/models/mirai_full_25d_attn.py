"""2.5D DBT model: per-slice 2D encoding + lightweight attention-based depth aggregation.

The model applies an existing 2D image encoder independently to each DBT slice
(shared weights across depth), then aggregates the per-slice feature vectors into
a single exam-level representation via attention pooling.

Key design choices:
- No 3D inflation of the encoder (inductive bias: 2D spatial encoding dominates).
- Depth aggregation is lightweight (linear attention score, no full transformer).
- Slices are encoded in small chunks along D to avoid GPU OOM.
- The central slice is not privileged; all slices compete equally for attention weight.
"""

import torch
import torch.nn as nn

from onconet.models.factory import load_model, RegisterModel, get_model_by_name


@RegisterModel("mirai_full_25d_attn")
class MiraiFull25dAttn(nn.Module):
    """2.5D DBT model with 2D per-slice encoding and attention-based depth pooling.

    Args (read from ``args``):
        img_encoder_snapshot (str | None): Path to a pre-trained 2D encoder
            snapshot.  If *None*, a ``custom_resnet`` is built from scratch.
        freeze_image_encoder (bool): If True, the encoder weights are frozen
            (``requires_grad=False``) and the forward pass is wrapped in
            ``torch.no_grad()``.
        slice_encoder_chunk_size (int): Number of slices processed together in
            one encoder forward pass.  Reduce this to save GPU memory.  Default 4.
        slice_attn_dropout (float): Dropout probability applied to the raw
            attention logits before softmax.  Default 0.0.
        slice_token_drop (float): Probability of randomly masking out each
            slice token **during training** before attention pooling.  Default 0.0.
        num_classes (int): Number of output classes.

    Forward:
        x: ``(B, D, C, H, W)`` — batch × depth-slices × channels × H × W.
        risk_factors: passed through but currently unused (kept for interface
            compatibility with other models).
        batch: unused, kept for interface compatibility.

    Returns:
        (logit, hidden, activ_dict) matching the interface of all other models.
    """

    def __init__(self, args):
        super(MiraiFull25dAttn, self).__init__()
        self.args = args

        # ------------------------------------------------------------------
        # 2D image encoder — NOT inflated to 3D
        # ------------------------------------------------------------------
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot, args, do_wrap_model=False
            )
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Feature dimension produced by the 2D encoder
        encoder_args = self.image_encoder._model.args
        self.image_repr_dim = getattr(
            encoder_args, "img_only_dim", encoder_args.hidden_dim
        )

        # ------------------------------------------------------------------
        # Chunk size for memory-efficient slice encoding
        # ------------------------------------------------------------------
        self.slice_encoder_chunk_size = getattr(
            args, "slice_encoder_chunk_size", 4
        )

        # ------------------------------------------------------------------
        # Lightweight depth aggregation via attention pooling
        # ------------------------------------------------------------------
        slice_attn_dropout = getattr(args, "slice_attn_dropout", 0.0)
        self.attn_score = nn.Linear(self.image_repr_dim, 1, bias=True)
        self.attn_dropout = nn.Dropout(p=slice_attn_dropout)

        # Optional slice-token drop rate (training only)
        self.slice_token_drop = getattr(args, "slice_token_drop", 0.0)

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        self.fc = nn.Linear(self.image_repr_dim, args.num_classes)

        # Expose image repr dim for downstream code that reads args.img_only_dim
        args.img_only_dim = self.image_repr_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_slices(self, x):
        """Encode a ``(B, D, C, H, W)`` volume slice by slice.

        Slices are processed in chunks of ``slice_encoder_chunk_size`` to
        control peak GPU memory usage.

        Returns:
            slice_feats: ``(B, D, F)`` per-slice feature vectors.
        """
        B, D, C, H, W = x.size()
        chunk_size = self.slice_encoder_chunk_size
        freeze = (
            hasattr(self.args, "freeze_image_encoder")
            and self.args.freeze_image_encoder
        )

        chunks = []
        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            n = end - start
            x_chunk = x[:, start:end].contiguous().view(B * n, C, H, W)

            if freeze:
                with torch.no_grad():
                    _, feat_chunk, _ = self.image_encoder(x_chunk)
            else:
                _, feat_chunk, _ = self.image_encoder(x_chunk)

            # Trim to image_repr_dim (handles risk-factor-augmented hidden dims)
            feat_chunk = feat_chunk.view(B, n, -1)[:, :, : self.image_repr_dim]
            chunks.append(feat_chunk)

        return torch.cat(chunks, dim=1)  # (B, D, F)

    def forward(self, x, risk_factors=None, batch=None):
        """
        Args:
            x: ``(B, D, C, H, W)``
        Returns:
            logit:      ``(B, num_classes)``
            hidden:     ``(B, F)``  — exam-level feature vector
            activ_dict: dict with keys ``'activ'`` and ``'attn_weights'``
        """
        B, D, C, H, W = x.size()

        # ---- per-slice encoding ----
        slice_feats = self._encode_slices(x)  # (B, D, F)

        # ---- optional slice-token dropout (training only) ----
        if self.training and self.slice_token_drop > 0.0:
            keep = torch.rand(B, D, device=x.device) > self.slice_token_drop
            # Guarantee at least one slice is kept per sample
            all_dropped = ~keep.any(dim=1)  # (B,)
            if all_dropped.any():
                keep[all_dropped, D // 2] = True
            attn_mask = keep  # (B, D)
        else:
            attn_mask = None

        # ---- attention pooling over depth ----
        attn_logits = self.attn_score(slice_feats).squeeze(-1)  # (B, D)
        attn_logits = self.attn_dropout(attn_logits)

        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, D)

        # Weighted sum → single exam-level feature
        hidden = (slice_feats * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, F)

        # ---- classification ----
        logit = self.fc(hidden)

        activ_dict = {
            "activ": slice_feats,
            "attn_weights": attn_weights,
        }

        return logit, hidden, activ_dict
