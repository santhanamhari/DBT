import sys
import os
import warnings
import hashlib
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from onconet.datasets.loader.dicom_multiframe import load_multiframe_dicom, normalize_minmax


CORRUPTED_FILE_ERR = 'Error: {}. Removing file from cache.'
DEFAULT_CACHE_EXT = '.png'


def save_slices_imwrite(x, path, mean=84.8727, std=93.9455, outdir="save_slices"):
    """Save each (3, H, W) slice in (T, 3, H, W) normalized tensor x as a PNG, unnormalizing first."""
    T, C, H, W = x.shape
    assert C == 3, "This function expects 3 channels (RGB)."
    os.makedirs(outdir, exist_ok=True)
    core_name = os.path.splitext(os.path.basename(path))[0]
    mean_arr = np.array([mean, mean, mean]).reshape(3, 1, 1)
    std_arr = np.array([std, std, std]).reshape(3, 1, 1)

    for t in range(T):
        img_norm = x[t].cpu().numpy()  # shape (3, H, W)
        img_unnorm = (img_norm * std_arr) + mean_arr  # unnormalize per-channel
        img_unnorm = np.clip(img_unnorm, 0, 255).astype(np.uint8)
        # Convert from (C, H, W) to (H, W, C) for cv2
        img_unnorm = np.transpose(img_unnorm, (1, 2, 0))
        fname = os.path.join(outdir, f"{core_name}_slice_{t:03d}.png")
        cv2.imwrite(fname, img_unnorm)


def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


class ComposeTrans:
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, image, additional=None):
        for t in self.transformers:
            image = t(image, additional)
        return image


def apply_transformers_and_cache(image, additional, path, transformers, cache,
                                 cache_full_size=True, base_key='default/'):
    composed = ComposeTrans(transformers)
    output = composed(image, additional)

    # Cache expects PIL-like objects that can .save()
    if cache is not None and hasattr(output, "save"):
        cache.add(path, base_key, output)
    elif cache is not None and cache_full_size and hasattr(image, "save"):
        cache.add(path, 'default/', image)

    return output


def split_transformers_by_cache(transformers):
    """Split transformer list into stages separated by cacheable transformers."""
    stages = []
    stage_key = 'default/'
    stage_trans = []

    for t in transformers:
        if hasattr(t, 'caching_keys'):
            stages.append((stage_key, stage_trans))
            stage_key = t.caching_keys()
            stage_trans = []
        else:
            stage_trans.append(t)

    stages.append((stage_key, stage_trans))
    return stages


class cache:
    def __init__(self, path, extension=DEFAULT_CACHE_EXT):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension

    def _file_dir(self, attr_key):
        return os.path.join(self.cache_dir, attr_key)

    def _file_path(self, attr_key, hashed_key):
        return os.path.join(self.cache_dir, attr_key, hashed_key + self.files_extension)

    def exists(self, image_path, attr_key):
        hashed_key = md5(image_path)
        return os.path.isfile(self._file_path(attr_key, hashed_key))

    def get(self, image_path, attr_key):
        hashed_key = md5(image_path)
        return Image.open(self._file_path(attr_key, hashed_key))

    def add(self, image_path, attr_key, image):
        hashed_key = md5(image_path)
        file_dir = self._file_dir(attr_key)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        image.save(self._file_path(attr_key, hashed_key))

    def rem(self, image_path, attr_key):
        hashed_key = md5(image_path)
        try:
            os.remove(self._file_path(attr_key, hashed_key))
        except OSError:
            pass


def _process_single_slice(args_tuple):
    """Worker function for parallel slice processing (non-consistent mode).

    Takes a tuple of (slice_tensor, transformers, additional).
    """
    slice_t, transformers, additional = args_tuple

    # Convert to uint8 for PIL
    frame_np = (slice_t * 255.0).astype(np.uint8)
    pil = Image.fromarray(frame_np, mode="L")

    composed = ComposeTrans(transformers)
    out = composed(pil, additional)
    return out


def _cuda_rng_is_safe() -> bool:
    """Avoid initializing CUDA just to snapshot RNG state."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        return False


def _get_rng_state() -> Dict[str, Any]:
    """Snapshot python/numpy/torch RNG state so we can restore later."""
    state: Dict[str, Any] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": None,
    }
    if _cuda_rng_is_safe():
        try:
            state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            state["cuda"] = None
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    """Restore python/numpy/torch RNG state."""
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and _cuda_rng_is_safe():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


def _seed_all(seed: int) -> None:
    """Seed python/numpy/torch RNGs."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32 - 1))
    torch.manual_seed(int(seed))
    if _cuda_rng_is_safe():
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass


class image_loader:
    def __init__(
        self,
        cache_path,
        transformers,
        num_workers: Optional[int] = None,
        *,
        consistent_across_slices: bool = True,
        args=None,
    ):
        """Image + DBT/volumetric slice loader.

        Args:
            cache_path: Path to a cache directory, or None.
            transformers: List of 2D transformers (PIL -> tensor, etc.).
            num_workers: Threadpool worker count for non-consistent slice processing.
            consistent_across_slices:
                If True, random transforms are synchronized so *every slice in a DBT
                volume* receives the same random augmentation parameters (e.g., same
                crop/flip). This is the common desired behavior for DBT.
            args: Parsed CLI args (Namespace or None). Provides num_slices,
                slice_policy, and slice_jitter.
        """
        self.transformers = transformers
        self.consistent_across_slices = consistent_across_slices
        self.args = args

        # Auto-detect optimal number of workers
        if num_workers is None:
            self.num_workers = min(multiprocessing.cpu_count(), 64)  # cap to avoid overhead
        else:
            self.num_workers = num_workers

        if cache_path is not None:
            self.use_cache = True
            self.cache = cache(cache_path)
            self.split_transformers = split_transformers_by_cache(transformers)
        else:
            self.use_cache = False
            self.composed_all_transformers = ComposeTrans(transformers)

    # -----------------------------
    # 2D image loading (existing)
    # -----------------------------
    def get_image(self, path, additional):
        if not self.use_cache:
            image = Image.open(path)
            return self.composed_all_transformers(image, additional)

        for key, post_transformers in self.split_transformers:
            if self.cache.exists(path, key):
                try:
                    image = self.cache.get(path, key)
                    image = apply_transformers_and_cache(
                        image,
                        additional,
                        path,
                        post_transformers,
                        self.cache,
                        cache_full_size=False,
                        base_key=key,
                    )
                    return image
                except Exception:
                    warnings.warn(CORRUPTED_FILE_ERR.format(sys.exc_info()[0]))
                    self.cache.rem(path, key)

        all_transformers = self.split_transformers[-1][1]
        image = Image.open(path)
        image = apply_transformers_and_cache(image, additional, path, all_transformers, self.cache)
        return image

    def get_images(self, paths, additionals):
        additionals += [None] * (len(paths) - len(additionals))
        images = [self.get_image(path, additional) for path, additional in zip(paths, additionals)]
        images = torch.stack(images)
        images = images.permute(1, 0, 2, 3)  # (C, Tviews, H, W)
        return images

    # -----------------------------
    # DICOM multi-frame as "batch of 2D slices"
    # -----------------------------
    def _select_slice_indices(self, num_frames, target_slices, policy, jitter=0):
        """Returns a list of indices into [0, num_frames).

        Supports policies:
        - ``grouped`` / ``grouped_7x3`` (default): stratified 7-bin × 3-adjacent-slice
          sampling.  When ``target_slices`` is 0 / None all frames are returned.
        - ``center_crop``, ``uniform``, ``pad``: legacy policies.

        Args:
            num_frames: Total number of slices in the volume.
            target_slices: Desired output slice count.  0 / None → all frames.
            policy: Name of selection policy.
            jitter: Max per-bin center offset sampled uniformly from
                ``[-jitter, +jitter]`` (training augmentation).
        """
        # Normalise "grouped_7x3" to canonical name
        if policy == "grouped_7x3":
            policy = "grouped"

        if target_slices is None or target_slices <= 0 or target_slices == num_frames:
            return list(range(num_frames))

        if policy == "grouped":
            return self._grouped_slice_indices(num_frames, target_slices, jitter)

        if num_frames > target_slices:
            if policy == "center_crop":
                start = (num_frames - target_slices) // 2
                return list(range(start, start + target_slices))
            if policy == "uniform":
                # evenly sample target_slices indices
                return np.linspace(0, num_frames - 1, target_slices).round().astype(int).tolist()
            raise ValueError(f"Unknown slice_policy for downsampling: {policy}")

        # num_frames < target_slices
        if policy == "pad":
            return list(range(num_frames))  # we'll pad later
        if policy in ("center_crop", "uniform"):
            # fallback: repeat last slice to reach target
            idx = list(range(num_frames))
            while len(idx) < target_slices:
                idx.append(idx[-1])
            return idx
        raise ValueError(f"Unknown slice_policy for upsampling: {policy}")

    def _grouped_slice_indices(self, num_frames, target_slices, jitter=0):
        """Stratified 7-bin × 3-adjacent-slice selection.

        Divides depth into ``num_bins = target_slices // 3`` equal-width bins.
        For each bin a center index is chosen (with optional jitter) and the
        three adjacent slices (center-1, center, center+1) are selected.  All
        indices are clamped to ``[0, num_frames-1]``.  Duplicate indices are
        kept to preserve a fixed output length equal to ``target_slices``.

        When ``target_slices % 3 != 0`` the last few bins select only the
        required number of extra slices (1 or 2 from the center).
        """
        num_bins = target_slices // 3
        remainder = target_slices % 3  # 0, 1, or 2

        indices = []
        for b in range(num_bins):
            # Bin spans [b/num_bins, (b+1)/num_bins] of the volume depth.
            bin_lo = b * num_frames / num_bins
            bin_hi = (b + 1) * num_frames / num_bins
            center = int(round((bin_lo + bin_hi) / 2.0))

            if jitter > 0:
                center += random.randint(-jitter, jitter)
            center = max(0, min(num_frames - 1, center))

            indices.extend([
                max(0, center - 1),
                center,
                min(num_frames - 1, center + 1),
            ])

        if remainder > 0:
            # Select from the tail portion of the volume that the regular bins did not cover.
            bin_lo = num_frames if num_bins > 0 else 0
            center = max(0, min(num_frames - 1, int(round((bin_lo + num_frames) / 2.0))))
            if jitter > 0:
                center += random.randint(-jitter, jitter)
            center = max(0, min(num_frames - 1, center))
            if remainder == 1:
                indices.append(center)
            else:
                indices.extend([max(0, center - 1), center])

        return indices

    def _process_slices_consistently(
        self,
        selected_slices: Sequence[np.ndarray],
        additional,
    ):
        """Apply the *same* (potentially random) 2D transform pipeline to each slice.

        Many augmentation pipelines include random operations (random crop, flip, etc.).
        If those are applied independently per slice, a DBT volume becomes spatially
        inconsistent slice-to-slice.

        This helper samples a single RNG seed per-volume and then resets RNG state to
        that seed before transforming *each* slice, guaranteeing identical random
        parameters across the whole volume.
        """
        # Sample once per-volume so augmentation can still vary volume-to-volume.
        # We sample the seed *before* snapshotting the outer RNG state so that this
        # one draw still advances global RNG (important for training diversity).
        volume_seed = int(torch.randint(0, 2 ** 32, (1,), dtype=torch.int64).item())

        outer_state = _get_rng_state()
        try:
            _seed_all(volume_seed)
            base_state = _get_rng_state()

            outs = []
            for slice_t in selected_slices:
                _set_rng_state(base_state)
                frame_np = (slice_t * 255.0).astype(np.uint8)
                pil = Image.fromarray(frame_np, mode="L")
                out = self.composed_all_transformers(pil, additional)
                outs.append(out)
        finally:
            _set_rng_state(outer_state)

        return outs

    def get_dicom_slices_as_2d_batch(self, path, additional):
        """Load a multi-frame DICOM as a stack of 2D slices.

        By default (consistent_across_slices=True), this applies the 2D transformer
        pipeline with *synchronized randomness* so every slice in the DBT volume gets
        the same random augmentation parameters.

        Returns:
            x: torch.Tensor (T, C, H, W)
        """
        if self.use_cache:
            raise NotImplementedError(
                "DICOM slice caching not implemented; run with --cache_path None"
            )

        vol = load_multiframe_dicom(path)
        vol = normalize_minmax(vol)  # (T, H, W)

        # Read slice control from args if present
        args = self.args
        target_slices = getattr(args, "num_slices", None) if args is not None else None
        policy = getattr(args, "slice_policy", "grouped") if args is not None else "grouped"
        jitter = getattr(args, "slice_jitter", 0) if args is not None else 0

        num_frames = vol.shape[0]
        idxs = self._select_slice_indices(num_frames, target_slices, policy, jitter=jitter)

        selected_slices = [vol[i] for i in idxs]

        # --- IMPORTANT CHANGE ---
        # In DBT, we often want the same spatial transform (crop/flip/etc.) applied to
        # every slice so the stack stays aligned.
        if self.consistent_across_slices:
            outs = self._process_slices_consistently(selected_slices, additional)
        else:
            # Non-consistent mode: keep previous parallel behavior.
            if len(selected_slices) > 1 and self.num_workers > 1:
                work_items = [(slice_t, self.transformers, additional) for slice_t in selected_slices]
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(_process_single_slice, item) for item in work_items]
                    outs = [future.result() for future in futures]
            else:
                outs = []
                for slice_t in selected_slices:
                    frame_np = (slice_t * 255.0).astype(np.uint8)
                    pil = Image.fromarray(frame_np, mode="L")
                    out = self.composed_all_transformers(pil, additional)
                    outs.append(out)

        x = torch.stack(outs, dim=0)  # (T, C, H, W)

        # Optional debug dump
        #try:
        #    save_slices_imwrite(x, path)
        #except Exception:
        #    # Don't crash training if debug visualization fails
        #    pass

        # If padding is requested and we had fewer real frames than target_slices, pad with zeros
        if (
            target_slices is not None
            and target_slices > 0
            and x.shape[0] < target_slices
            and policy == "pad"
        ):
            pad_n = target_slices - x.shape[0]
            pad = torch.zeros((pad_n, *x.shape[1:]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        return x

    def get_dicom_volumes_as_2d_batches(self, paths, additionals):
        """Returns:
        vols: torch.Tensor (Nviews, T, C, H, W) with fixed T if args.num_slices set.
        """
        additionals += [None] * (len(paths) - len(additionals))
        vols = [self.get_dicom_slices_as_2d_batch(p, a) for p, a in zip(paths, additionals)]
        return torch.stack(vols, dim=0)
