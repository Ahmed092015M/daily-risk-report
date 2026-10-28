"""
Fund configuration: read config/config.yaml and check it before anything else runs.
"""
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["load_config", "asset_table"]

#REQUIRED = ["fund", "assets", "fred", "overlay", "models", "measures", "limits", "stress"]
REQUIRED = ["fund", "assets", "market", "fred", "overlay", "models", "measures", "limits", "stress"]


def load_config(path):
    """Read the YAML configuration and validate it.

    Parameters
    ----------
    path : str or Path
        Location of config.yaml.

    Returns
    -------
    dict
        The configuration, unchanged, after the checks below.

    Raises
    ------
    ValueError
        If a section is missing, a currency is not EUR/USD, or the weights do not sum to 1.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config: missing sections {missing}")

    for key, a in cfg["assets"].items():
        if a["currency"] not in ("EUR", "USD"):
            raise ValueError(f"config: asset {key} has currency {a['currency']} (only EUR and USD)")

    total = sum(a["weight"] for a in cfg["assets"].values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"config: target weights sum to {total:.4f}, not 1")

    if cfg["overlay"]["enabled"] and cfg["overlay"]["underlying"] not in cfg["assets"]:
        raise ValueError(f"config: overlay underlying {cfg['overlay']['underlying']} is not an asset")
    return cfg


def asset_table(cfg):
    """The assets as a DataFrame (one row per asset), for display and for the code that follows.

    Returns
    -------
    pd.DataFrame
        Index = asset key; columns ticker, name, currency, asset_class, weight and the sensitivities
        (NaN where not given).
    """
    return pd.DataFrame(cfg["assets"]).T.infer_objects()