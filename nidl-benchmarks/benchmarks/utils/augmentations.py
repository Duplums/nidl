##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from nidl.transforms import MultiViewsTransform
from nidl.volume.transforms.augmentation import (
    RandomErasing,
    RandomGaussianNoise,
    RandomResizedCrop,
)
from nidl.volume.transforms.preprocessing import CropOrPad, ZNormalization
from torchvision.transforms import Compose


def simclr_augmentations(
    erase_min: float,
    erase_max: float,
    crop_size: int,
    crop_min: float,
    crop_max: float,
):
    return MultiViewsTransform(
        Compose(
            [
                RandomErasing(scale=(erase_min, erase_max), p=0.8),
                RandomResizedCrop(
                    target_shape=crop_size, scale=(crop_min, crop_max), p=0.8
                ),
                CropOrPad(target_shape=crop_size),
                ZNormalization(),
                RandomGaussianNoise(p=0.2),
            ]
        ),
        n_views=2,
    )


def dino_augmentations(
    global_crop_size: int,
    global_crop_min: float,
    global_crop_max: float,
    local_crop_size: int,
    local_crop_min: float,
    local_crop_max: float,
    num_local_crops: int,
    num_global_crops: int = 2,
):
    # Global crops
    global_tf = [
        Compose(
            [
                RandomResizedCrop(
                    global_crop_size, scale=(global_crop_min, global_crop_max)
                ),
                CropOrPad(target_shape=global_crop_size),
                RandomGaussianNoise(p=0.2),
                ZNormalization(),
            ]
        )
        for _ in range(num_global_crops)
    ]
    # Local crops
    local_tf = [
        Compose(
            [
                RandomResizedCrop(
                    local_crop_size, scale=(local_crop_min, local_crop_max)
                ),
                CropOrPad(target_shape=local_crop_size),
                RandomGaussianNoise(p=0.2),
                ZNormalization(),
            ]
        )
        for _ in range(num_local_crops)
    ]
    return MultiViewsTransform(global_tf + local_tf)


def ijepa_augmentations(crop_size: int, scale_min: float, scale_max: float):
    return Compose(
        [
            RandomResizedCrop(crop_size, scale=(scale_min, scale_max)),
            ZNormalization(),
        ]
    )
