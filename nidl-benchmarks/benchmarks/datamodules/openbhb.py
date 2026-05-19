##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Callable, Optional, Union

from nidl.datasets import OpenBHB
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class OpenBHBDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        modality: str,
        batch_size: int,
        num_workers: int,
        target: Union[str, list[str]] = "age",
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.root = root
        self.modality = modality
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # NIDL transforms
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.target_transform = target_transform

        dataset_kwargs = {
            "root": self.root,
            "modality": self.modality,
            "target": self.target,
            "target_transforms": self.target_transform,
        }

        self.train_set = OpenBHB(
            split="train", transforms=self.train_transform, **dataset_kwargs
        )
        self.val_set = OpenBHB(
            split="val", transforms=self.val_transform, **dataset_kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
