"""
Config loader. Every experiment script pulls settings from
config/default_config.yaml through this module rather than
redefining defaults locally — this was one of the main sources
of drift in the original code (e.g. n_qubits=20 in one file,
20-qubit assumed but not enforced in another).
"""
import os
import yaml

_CONFIG_CACHE = None


_DATA_PATH_KEYS = ("dukascopy_file", "forexsb_file", "fred_file", "processed_dir")


def _anchor_data_paths(cfg: dict, config_path: str) -> dict:
    """Make relative data paths absolute, anchored at the repo root (the folder holding config/).

    Without this, "data/raw/x.csv" resolved against the CURRENT WORKING DIRECTORY, so a script launched from
    any other folder wrote the downloaded data into <that folder>/data/raw and the rest of the pipeline could
    not find it. Absolute paths in the YAML are left untouched.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    data = cfg.get("data")
    if isinstance(data, dict):
        for key in _DATA_PATH_KEYS:
            val = data.get(key)
            if isinstance(val, str) and val and not os.path.isabs(val):
                data[key] = os.path.normpath(os.path.join(root, val))
    return cfg


def load_config(path: str = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "default_config.yaml",
        )

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _anchor_data_paths(cfg, path)

    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def merge_override(base_cfg: dict, override: dict) -> dict:
    """Shallow-merge an override dict (e.g. for ablations) into a base config section."""
    merged = dict(base_cfg)
    merged.update(override)
    return merged
