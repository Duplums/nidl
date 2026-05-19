##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def save_config(cfg, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, f"{log_dir}/config.yaml")
    return log_dir


@hydra.main(version_base=None, config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    save_config(cfg, cfg.pretrain.log_dir)

    logger = TensorBoardLogger(
        save_dir=cfg.pretrain.log_dir,
        name="",
    )

    datamodule = instantiate(cfg.pretrain.datamodule)

    datamodule.setup(stage="fit")

    estimator = instantiate(
        cfg.pretrain.ssl,
        callbacks=instantiate(cfg.pretrain.callbacks),
        logger=logger,
    )

    estimator.fit(datamodule.train_dataloader(), datamodule.val_dataloader())


if __name__ == "__main__":
    main()
