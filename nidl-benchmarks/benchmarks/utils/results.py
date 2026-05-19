##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
import json
from pathlib import Path


def save_results(model_name: str, task: str, metrics: dict):
    out_dir = Path(__file__).parents[2] / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"

    existing = json.loads(out_file.read_text()) if out_file.exists() else {}
    existing[task] = metrics
    out_file.write_text(json.dumps(existing, indent=2))
    print(f"Results saved → {out_file}")