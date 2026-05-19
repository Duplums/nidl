##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
import numpy as np
from nidl.volume.transforms.preprocessing import CropOrPad, ZNormalization
from torchvision.transforms import Compose, Lambda


def inference_transform(crop_size: int):
    def squeeze_leading_then_unsqueeze_first(arr, dtype=np.float32):
        # Remove all leading dimensions of size 1
        while arr.ndim > 0 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        # Add one channel dimension at the front
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(dtype=dtype)

    return Compose(
        [
            Lambda(squeeze_leading_then_unsqueeze_first),
            CropOrPad(crop_size),
            ZNormalization(),
        ]
    )
