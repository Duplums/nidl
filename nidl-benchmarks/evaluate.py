"""
python evaluate.py +pretrain=simclr_resnet50_openbhb +downstream=age_regression

# All tasks at once:
python evaluate.py +pretrain=simclr_resnet50_openbhb \
    +downstream=age_regression,clinical_diagnosis --multirun
"""

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error


@hydra.main(config_path="configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Load pretrained backbone
    backbone = instantiate(cfg.pretrain.backbone)
    model = instantiate(cfg.pretrain.ssl, encoder=backbone)
    ckpt = f"results/{cfg.pretrain.model_name}/checkpoint.pt"
    model.load_state_dict(torch.load(ckpt))

    # Embed — reuses nidl's .transform()
    dataset = instantiate(cfg.pretrain.dataset)  # no augmentations
    Z_train = model.transform(dataset.train_dataloader()).cpu().numpy()
    Z_test = model.transform(dataset.val_dataloader()).cpu().numpy()
    y_train = dataset.train_labels()
    y_test = dataset.val_labels()

    # Probe
    probe = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Z_train, y_train)
    y_pred = probe.predict(Z_test)
    metrics = {"mae": round(mean_absolute_error(y_test, y_pred), 3)}

    # Persist
    out = Path(f"results/{cfg.pretrain.model_name}/metrics.json")
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[cfg.downstream.task] = metrics
    out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
