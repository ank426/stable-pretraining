"""SimDINOv2 masked-patch machinery.

SimDINOv2 pairs DINOv2's CLS-token self-distillation with iBOT-style masked-patch
prediction. The patch term hides a set of patches from the student, and this
module decides *which* patches those are.

**Polarity.** ``True`` means the patch is **masked out** (hidden from the
student). This is the opposite of the ``1 = in block = visible`` convention used
by :func:`stable_pretraining.data.masking.multi_block_mask`, which serves
SALT, I-JEPA, and the MAE-style transforms. The two must never be interchanged;
the polarity here is fixed by :class:`~stable_pretraining.losses.SimDINOv2Loss`,
which expects a mask of hidden positions.

These helpers are private to the method. Other methods in this codebase keep
their own masking inline -- see the ``_random_mask`` helpers in ``ibot``,
``dinov2``, and ``msn`` -- and none of them share an implementation, so this
stays local until a second method actually needs it.
"""

import math

import torch


class _UniformPool:
    """Replenishable stream of floats drawn from one ``torch.Generator``.

    Drawing the whole batch's randomness up front is what keeps mask generation
    off the critical path: the block-placement loop is inherently sequential per
    row, but every random decision it makes can be vectorized in a single call.
    Measured on a 14x14 grid, drawing per-decision costs ~55 ms per step for 256
    rows versus ~25 ms for this pooled form.
    """

    def __init__(self, generator: torch.Generator | None, size: int):
        self.generator = generator
        self.values = self._draw(size)

    def _draw(self, size: int) -> list[float]:
        return torch.rand(size, generator=self.generator).tolist()

    def next(self) -> float:
        if not self.values:
            # Refill unconditionally rather than imposing a draw budget. The
            # placement loop is bounded by construction -- it stops at the
            # first zero-delta attempt and otherwise raises ``count`` by at
            # least one per outer pass -- so refilling cannot spin forever, and
            # capping it here would only introduce failures upstream never has.
            self.values = self._draw(1024)
        return self.values.pop()


def _place_blocks(
    height: int,
    width: int,
    target: int,
    min_num_patches: int,
    max_num_patches: int,
    log_aspect_ratio: tuple[float, float],
    pool: _UniformPool,
) -> int:
    """Accumulate rectangular blocks into a Python-int bitmask.

    Faithful to ``simdinov2/data/masking.py``: each attempt draws a target area
    and an aspect ratio, converts them to a ``(h, w)`` block that must fit with a
    strict inequality on both axes, and is accepted only when it covers at least
    one previously-unmasked cell without exceeding the remaining budget. Ten
    attempts are made per block, and the first accepted block ends them.

    Args:
        height: Grid height in patches.
        width: Grid width in patches.
        target: Desired number of masked cells; placement stops early on ``0``.
        min_num_patches: Lower bound of a block's target area.
        max_num_patches: Upper bound of a block's target area.
        log_aspect_ratio: Natural-log aspect ratio bounds, sampled uniformly.
        pool: Source of uniform ``[0, 1)`` floats.

    Returns:
        int: Bitmask over ``height * width`` cells, bit ``i * width + j`` set when
        cell ``(i, j)`` is masked.
    """
    n_cells = height * width
    lo, hi = log_aspect_ratio
    bits = 0
    count = 0

    while count < target:
        budget = min(target - count, max_num_patches)
        delta = 0
        for _ in range(10):
            area = pool.next() * (budget - min_num_patches) + min_num_patches
            aspect_ratio = math.exp(pool.next() * (hi - lo) + lo)
            h = int(round(math.sqrt(area * aspect_ratio)))
            w = int(round(math.sqrt(area / aspect_ratio)))

            if w < width and h < height:
                top = int(pool.next() * (height - h + 1))
                left = int(pool.next() * (width - w + 1))
                row_bits = ((1 << w) - 1) << left
                block = 0
                for row in range(h):
                    block |= row_bits << ((top + row) * width)

                fresh = (~bits) & block
                n_fresh = bin(fresh).count("1")
                if 0 < n_fresh <= budget:
                    bits |= fresh
                    delta = n_fresh
            if delta > 0:
                break

        if delta == 0:
            break
        count += delta

    assert bits < (1 << n_cells), "bitmask overflowed the grid"
    return bits


def _bits_to_row(bits: int, n_cells: int) -> list[int]:
    """Unpack a bitmask into a row-major list of ``n_cells`` 0/1 values.

    Single source of truth for bit layout: cell ``c`` lives at bit ``c``, where
    ``c = row * width + col``. Anything that unpacks a bitmask must go through
    here -- re-deriving the index arithmetic is how a driver ends up reading
    every ``width``-th bit.
    """
    return [(bits >> cell) & 1 for cell in range(n_cells)]


def _bits_to_mask(bits: int, height: int, width: int) -> torch.Tensor:
    """Expand a ``height * width`` bitmask into a ``[height, width]`` bool tensor."""
    cells = _bits_to_row(bits, height * width)
    return torch.tensor(
        [cells[row * width : (row + 1) * width] for row in range(height)],
        dtype=torch.bool,
    )


def _log_aspect_bounds(min_aspect: float, max_aspect: float | None):
    if min_aspect <= 0:
        raise ValueError(f"min_aspect must be positive, got {min_aspect}")
    if max_aspect is None:
        max_aspect = 1 / min_aspect
    if max_aspect <= 0:
        raise ValueError(f"max_aspect must be positive, got {max_aspect}")
    if max_aspect < min_aspect:
        raise ValueError(
            f"max_aspect ({max_aspect}) must be >= min_aspect ({min_aspect})"
        )
    return math.log(min_aspect), math.log(max_aspect)


def _simdinov2_mask(
    grid_size: tuple[int, int],
    num_masking_patches: int = 0,
    min_num_patches: int = 4,
    max_num_patches: int | None = None,
    min_aspect: float = 0.3,
    max_aspect: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate one rectangular block mask.

    Ports the official ``simdinov2`` ``MaskingGenerator``: place axis-aligned
    rectangular blocks until the requested count is reached, rejecting any block
    that fails to cover a fresh cell or that overshoots the remaining budget.

    Placement stops as soon as a block attempt adds nothing, so **masked counts
    routinely undershoot the requested target**. That is faithful to upstream and
    not a failure. Use :func:`_simdinov2_patch_masks` for the batched, stratified
    form the method actually consumes.

    Args:
        grid_size: Patch grid as ``(height, width)``.
        num_masking_patches: Target number of masked patches. ``0`` returns an
            all-false mask, which is how unmasked rows are represented.
        min_num_patches: Lower bound of a block's target area. Default is 4.
        max_num_patches: Upper bound of a block's target area. Defaults to
            ``num_masking_patches``.
        min_aspect: Minimum block aspect ratio (height/width). Default is 0.3.
        max_aspect: Maximum block aspect ratio. Defaults to ``1 / min_aspect``.
        generator: Optional ``torch.Generator`` for reproducible masks.

    Returns:
        torch.Tensor: ``[height, width]`` bool tensor, ``True`` where masked.

    Raises:
        ValueError: If the grid is smaller than 2x2, or the block bounds are
            inconsistent.

    Example:
        >>> mask = _simdinov2_mask((14, 14), 20)
        >>> mask.shape
        torch.Size([14, 14])
        >>> mask.dtype == torch.bool
        True
        >>> int(mask.sum()) <= 20
        True
        >>> bool(_simdinov2_mask((14, 14), 0).any())
        False
    """
    height, width = grid_size
    if height < 2 or width < 2:
        raise ValueError(f"grid must be at least 2x2 to hold a block, got {grid_size}")
    if min_num_patches < 1:
        raise ValueError(f"min_num_patches must be >= 1, got {min_num_patches}")
    log_aspect_ratio = _log_aspect_bounds(min_aspect, max_aspect)

    # Resolve ``max_num_patches`` only after the zero-target exit: it defaults
    # to the target, so a target of 0 would otherwise trip the block-bound
    # check below and turn "no masks requested" into a spurious ValueError.
    if num_masking_patches <= 0:
        return torch.zeros((height, width), dtype=torch.bool)

    if max_num_patches is None:
        max_num_patches = num_masking_patches
    if max_num_patches < min_num_patches:
        raise ValueError(
            f"max_num_patches ({max_num_patches}) must be >= "
            f"min_num_patches ({min_num_patches})"
        )

    bits = _place_blocks(
        height,
        width,
        num_masking_patches,
        min_num_patches,
        max_num_patches,
        log_aspect_ratio,
        _UniformPool(generator, 8),
    )
    return _bits_to_mask(bits, height, width)


def _simdinov2_patch_masks(
    n_rows: int,
    n_patches: int,
    grid_size: tuple[int, int],
    mask_ratio_min_max: tuple[float, float] = (0.1, 0.5),
    mask_sample_probability: float = 0.5,
    max_num_patches: int | None = None,
    min_num_patches: int = 4,
    min_aspect: float = 0.3,
    max_aspect: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Assign per-row block masks the way SimDINOv2's collate function does.

    Each row is one ``(view, sample)`` pair. Only ``int(n_rows *
    mask_sample_probability)`` rows receive a mask, their target ratios are
    **stratified** across ``mask_ratio_min_max`` so no two rows draw from the
    same band, the remaining rows get empty masks, and finally all rows are
    permuted -- so masked rows are scattered across views rather than
    front-loaded.

    Args:
        n_rows: Number of ``(view, sample)`` rows. Upstream uses
            ``n_global_crops * batch_size``, pooling every view together.
        n_patches: Patches per row, i.e. ``grid_h * grid_w``.
        grid_size: Patch grid as ``(height, width)``.
        mask_ratio_min_max: ``(min, max)`` masked-patch ratio band. Default is
            ``(0.1, 0.5)``, the official value.
        mask_sample_probability: Fraction of rows that receive a mask. Default is
            0.5, the official value.
        max_num_patches: Cap on a single block's area. Defaults to
            ``int(0.5 * n_patches)``.
        min_num_patches: Lower bound of a block's target area. Default is 4.
        min_aspect: Minimum block aspect ratio. Default is 0.3.
        max_aspect: Maximum block aspect ratio. Defaults to ``1 / min_aspect``.
        generator: Optional ``torch.Generator`` for reproducible masks.

    Returns:
        torch.Tensor: ``[n_rows, n_patches]`` bool tensor, ``True`` where masked.
        Rows are ordered row-major over ``(view, sample)`` to match the
        ``flatten(0, 1)`` feature layout
        :class:`~stable_pretraining.losses.SimDINOv2Loss` expects.

    Raises:
        ValueError: If the row count, grid or ratio/probability arguments are
            inconsistent.

    Example:
        >>> masks = _simdinov2_patch_masks(n_rows=16, n_patches=196, grid_size=(14, 14))
        >>> masks.shape
        torch.Size([16, 196])
        >>> masks.dtype == torch.bool
        True
        >>> int(masks.any(dim=1).sum()) <= 8
        True
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    if grid_size[0] * grid_size[1] != n_patches:
        raise ValueError(
            f"grid_size {grid_size} has {grid_size[0] * grid_size[1]} cells but "
            f"n_patches={n_patches}"
        )
    ratio_min, ratio_max = mask_ratio_min_max
    if not 0 < ratio_min <= ratio_max <= 1:
        raise ValueError(
            f"mask_ratio_min_max must satisfy 0 < min <= max <= 1, got {mask_ratio_min_max}"
        )
    if not 0.0 <= mask_sample_probability <= 1.0:
        raise ValueError(
            f"mask_sample_probability must be in [0, 1], got {mask_sample_probability}"
        )
    if min_num_patches < 1:
        raise ValueError(f"min_num_patches must be >= 1, got {min_num_patches}")
    if max_num_patches is None:
        max_num_patches = int(0.5 * n_patches)
    if max_num_patches < 1:
        raise ValueError(f"max_num_patches must be >= 1, got {max_num_patches}")
    aspect_bounds = _log_aspect_bounds(min_aspect, max_aspect)

    n_masked_rows = int(n_rows * mask_sample_probability)
    # One shared pool for the whole batch; placement stays sequential per row
    # because each block depends on what previous blocks already covered.
    pool = _UniformPool(generator, max(64, n_rows * 16))

    # Stratify so every masked row draws its ratio from its own sub-band.
    bands = torch.linspace(ratio_min, ratio_max, n_masked_rows + 1).tolist()
    rows = []
    for index in range(n_rows):
        if index < n_masked_rows:
            low, high = bands[index], bands[index + 1]
            target = int(n_patches * (pool.next() * (high - low) + low))
        else:
            target = 0
        rows.append(
            _place_blocks(
                grid_size[0],
                grid_size[1],
                target,
                min_num_patches,
                max_num_patches,
                aspect_bounds,
                pool,
            )
        )

    # Upstream shuffles the mask list before stacking, which is the same as
    # stacking then permuting: it decides *which* rows get masked, not what
    # shape those masks have.
    order = torch.randperm(n_rows, generator=generator).tolist()
    return torch.tensor(
        [_bits_to_row(rows[i], n_patches) for i in order],
        dtype=torch.bool,
    )
