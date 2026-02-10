##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Literal, Union

import numpy as np
import torch.nn.functional as func

from .....transforms import TypeTransformInput, VolumeTransform

PadMode = Literal["constant", "reflect", "replicate"]


class RandomCrop(VolumeTransform):
    """Randomly crop a fixed-size 3D patch from a volume.

    This transform is a 3D counterpart of common 2D random cropping
    augmentations and follows the practical behavior in `torchvision`:
    if the requested crop is larger than the input along any dimension, the
    input is first *padded* symmetrically and then the crop is sampled.
    This guarantees a valid crop for any input size while keeping the sampling
    procedure simple and reproducible.

    It handles :class:`numpy.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type).

    Parameters
    ----------
    target_shape : int or tuple of (int, int, int)
        Spatial shape of the output patch :math:`(H, W, D)`. If an int is
        provided, the same size is applied across all spatial dimensions.
    pad_if_needed : bool, default=True
        If True, pad the input volume if it's smaller than ``target_shape`` in
        any spatial dimension to avoid raising an exception.
    padding_mode : {'constant', 'reflect', 'replicate'}, default='constant'
        Padding strategy used when the input is smaller than ``target_shape``
        in any spatial dimension. It is applied before cropping.

        - ``'constant'`` pads with ``padding_value``.
        - ``'reflect'`` pads by reflecting the border values.
        - ``'replicate'`` pads by replicating the border values.
    padding_value : float, default=0.0
        Fill value used when ``padding_mode='constant'``.
    kwargs : dict
        Keyword arguments given to :class:`~nidl.transforms.VolumeTransform`.

    Notes
    -----
    **Input shapes.**
    This transform expects a 3D volume with shape :math:`(H, W, D)` or a
    channel-first volume with shape :math:`(C, H, W, D)`. The crop is sampled
    once and applied consistently across channels.

    **Sampling.**
    For each spatial dimension, the crop start index is sampled uniformly
    from the valid range. If the input has been padded, the sampling range
    is computed on the padded shape.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((1, 64, 64, 32))  # (C, H, W, D)
    >>> t = RandomCrop((48, 48, 24))
    >>> y = t(x)
    >>> y.shape
    (1, 48, 48, 24)

    """

    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        pad_if_needed: bool = True,
        padding_mode: PadMode = "constant",
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = self._parse_shape(target_shape, length=3)
        self.pad_if_needed = bool(pad_if_needed)
        self.padding_mode = padding_mode
        self.padding_value = float(padding_value)

        if self.padding_mode not in ("constant", "reflect", "replicate"):
            raise ValueError(
                "padding_mode must be in {'constant', 'reflect', 'replicate'}."
            )

    @staticmethod
    def _compute_required_padding(
        in_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Compute symmetric padding for each spatial dimension.

        Returns padding as ((pad_before_H, pad_after_H), (pad_before_W, ...),
        (pad_before_D, ...)).
        """
        pads = []
        for s_in, s_tgt in zip(in_shape, target_shape):
            missing = max(0, s_tgt - s_in)
            before = missing // 2
            after = missing - before
            pads.append((before, after))
        return pads[0], pads[1], pads[2]

    @staticmethod
    def _pad_numpy(
        data: np.ndarray,
        pads_hwd: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: PadMode,
        value: float,
        has_channels: bool,
    ) -> np.ndarray:
        """Pad a numpy volume to satisfy the target crop size."""
        (ph0, ph1), (pw0, pw1), (pd0, pd1) = pads_hwd
        if has_channels:
            pad_width = ((0, 0), (ph0, ph1), (pw0, pw1), (pd0, pd1))
        else:
            pad_width = ((ph0, ph1), (pw0, pw1), (pd0, pd1))

        if mode == "constant":
            return np.pad(
                data,
                pad_width=pad_width,
                mode="constant",
                constant_values=value,
            )

        # numpy names differ slightly from torch semantics; 'edge' matches
        # 'replicate' in torch
        if mode == "replicate":
            return np.pad(data, pad_width=pad_width, mode="edge")
        else:
            return np.pad(data, pad_width=pad_width, mode="reflect")

    @staticmethod
    def _pad_torch(
        data,
        pads_hwd: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: PadMode,
        value: float,
        has_channels: bool,
    ):
        """Pad a torch volume to satisfy the target crop size.

        Notes
        -----
        :func:`~torch.nn.functional.pad` expects padding in reverse order:
        (pad_left_D, pad_right_D, pad_left_W, pad_right_W, pad_left_H,
        pad_right_H).
        """
        (ph0, ph1), (pw0, pw1), (pd0, pd1) = pads_hwd
        pad = (pd0, pd1, pw0, pw1, ph0, ph1)

        if mode == "constant":
            return func.pad(data, pad=pad, mode="constant", value=float(value))
        else:
            return func.pad(data, pad=pad, mode=mode)

    @staticmethod
    def _random_crop_slices(
        in_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> list[slice]:
        """Sample crop slices uniformly at random."""
        slices = []
        for s_in, s_tgt in zip(in_shape, target_shape):
            if s_in == s_tgt:
                start = 0
            else:
                start = int(np.random.randint(0, s_in - s_tgt + 1))
            slices.append(slice(start, start + s_tgt))
        return slices

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Pad (if needed) and randomly crop the input volume.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input volume with shape :math:`(H, W, D)` or :math:`(C, H, W, D)`.

        Returns
        -------
        np.ndarray or torch.Tensor
            Randomly cropped patch with shape :math:`(H_t, W_t, D_t)` or
            :math:`(C, H_t, W_t, D_t)`, where :math:`(H_t, W_t, D_t)` equals
            ``target_shape``.

        Raises
        ------
        ValueError
            If input is not 3D or 4D channel-first.

        """
        if data.ndim not in (3, 4):
            raise ValueError(
                "RandomCrop expects data with shape (H, W, D) or (C, H, W, D)."
            )

        has_channels = data.ndim == 4
        spatial_shape = tuple(
            int(s) for s in (data.shape[1:] if has_channels else data.shape)
        )

        # Pad if necessary (torchvision-like behavior: pad then crop)
        pads_hwd = self._compute_required_padding(
            spatial_shape, self.target_shape
        )
        needs_padding = any(p0 > 0 or p1 > 0 for (p0, p1) in pads_hwd)

        if needs_padding:
            if isinstance(data, np.ndarray):
                data = self._pad_numpy(
                    data,
                    pads_hwd=pads_hwd,
                    mode=self.padding_mode,
                    value=self.padding_value,
                    has_channels=has_channels,
                )
            else:
                data = self._pad_torch(
                    data,
                    pads_hwd=pads_hwd,
                    mode=self.padding_mode,
                    value=self.padding_value,
                    has_channels=has_channels,
                )

        # Sample crop on (possibly padded) shape
        spatial_shape = tuple(
            int(s) for s in (data.shape[1:] if has_channels else data.shape)
        )
        crop_slices = self._random_crop_slices(
            spatial_shape, self.target_shape
        )

        if has_channels:
            crop_slices = [slice(None), *crop_slices]

        return data[tuple(crop_slices)]
