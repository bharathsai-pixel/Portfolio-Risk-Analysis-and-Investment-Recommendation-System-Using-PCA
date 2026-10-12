"""Reproducibility metadata and source-isolated output paths."""
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import sklearn


def source_paths(source, root="data"):
    if source not in {"yahoo_finance", "synthetic"}:
        raise ValueError("source must be 'yahoo_finance' or 'synthetic'")
    base = Path(root)
    return base / source / "raw_prices.csv", base / "processed" / source


def create_provenance(source, prices, start, end, tickers, seed=None, config=None):
    return {
        "provenance_id": str(uuid.uuid4()), "data_source": source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": str(start), "requested_end": str(end),
        "actual_start": prices.index.min().isoformat(), "actual_end": prices.index.max().isoformat(),
        "tickers": list(map(str, tickers)), "frequency": "daily trading observations",
        "missing_data_handling": "forward fill prices; rows incomplete across retained tickers dropped; feature NaNs mean-imputed",
        "synthetic_random_seed": seed if source == "synthetic" else None,
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "configuration": config or {},
    }


def save_provenance(metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
