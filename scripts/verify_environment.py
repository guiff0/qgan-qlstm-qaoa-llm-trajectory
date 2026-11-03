"""
Run this FIRST, before anything else. Checks that the environment is
actually ready, and tells you specifically what's missing rather than
failing partway through a multi-hour run.

Run with:  python -m scripts.verify_environment [--env-only]

--env-only skips the raw/processed data-file checks (use right after rebuilding the venv).
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(description: str, fn):
    try:
        result = fn()
        if result is False:
            CHECKS_FAILED.append(description)
            print(f"  [FAIL] {description}")
        else:
            CHECKS_PASSED.append(description)
            print(f"  [ OK ] {description}" + (f" — {result}" if isinstance(result, str) else ""))
    except Exception as e:
        CHECKS_FAILED.append(f"{description}: {e}")
        print(f"  [FAIL] {description} — {type(e).__name__}: {e}")
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last = tb[-1]
            print(f"         raised at {last.filename}:{last.lineno} in {last.name}")


def main():
    env_only = "--env-only" in sys.argv
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70)

    print("\n-- Python / packages --")
    check("Python >= 3.10", lambda: sys.version_info >= (3, 10))
    for pkg in ["torch", "pennylane", "numpy", "pandas", "scipy", "sklearn",
                "statsmodels", "yaml", "requests"]:
        check(f"Package installed: {pkg}", lambda p=pkg: importlib.import_module(p) is not None)

    # Needed by scripts/acquire_all_data.py and scripts/prepare_data.py. These were
    # previously unchecked, so a missing one only surfaced partway through a run.
    print("\n-- Data-acquisition / preparation packages --")
    for pkg in ["yfinance", "histdata", "psutil"]:
        check(f"Package installed: {pkg}", lambda p=pkg: importlib.import_module(p) is not None)

    # dukascopy_python raises NameError("requests") inside the package (a bug in the package itself, not a
    # missing dependency, so `pip install requests` cannot fix it). scripts/acquire_all_data.py falls back to
    # its own datafeed client, so this is a warning, not a failure.
    try:
        importlib.import_module("dukascopy_python")
        print("  [ OK ] Package installed: dukascopy_python")
    except Exception as e:
        print(f"  [WARN] dukascopy_python unusable ({type(e).__name__}: {e}); "
              "acquire_all_data.py uses its direct Dukascopy client instead (not a failure)")

    def numpy_pennylane_compat():
        import numpy
        try:
            import pennylane
        except ImportError:
            return "skipped (pennylane not installed; reported above)"
        if int(numpy.__version__.split(".")[0]) < 2:
            return (f"NumPy {numpy.__version__} with PennyLane {pennylane.__version__}: recent PennyLane "
                    f"deprecates NumPy<2 (warning only for now); `pip install -U numpy` when convenient")
        return f"NumPy {numpy.__version__}"
    check("NumPy / PennyLane compatibility (advisory)", numpy_pennylane_compat)

    print("\n-- Compute --")
    def gpu_check():
        import torch
        if torch.cuda.is_available():
            return f"CUDA available ({torch.cuda.get_device_name(0)})"
        return ("No CUDA GPU detected. Classical LSTM/GAN training will be slow on CPU; "
                "the QGAN's quantum simulator (default.qubit) is CPU-only regardless, "
                "but a GPU is strongly recommended for the classical components at "
                "5.5M-row scale. See RUNNING.md for cloud-GPU options.")
    check("GPU availability", gpu_check)

    def resources_check():
        import shutil
        import psutil
        ram = psutil.virtual_memory().total / 2**30
        free = shutil.disk_usage(".").free / 2**30
        msg = f"{ram:.0f} GiB RAM, {free:.0f} GiB free disk"
        if ram < 16:
            msg += " -- LOW: prepare_data.py holds several copies of a ~5M x 60 float64 frame"
        if free < 10:
            msg += " -- LOW: raw CSVs + caches + processed arrays need several GiB"
        return msg
    check("Memory / disk (advisory)", resources_check)

    print("\n-- Config --")
    def config_check():
        from src.utils.config import load_config
        cfg = load_config()
        return f"Loaded, {len(cfg)} top-level sections"
    check("config/default_config.yaml loads", config_check)

    try:
        from src.utils.config import load_config
        cfg = load_config()
        cfg["data"]
    except Exception as e:  # noqa: BLE001 - report it in the summary instead of a traceback
        CHECKS_FAILED.append(f"Cannot read data paths from config: {e}")
        print(f"\n  [FAIL] Cannot read data paths from config: {e}")
        cfg = None

    if cfg is not None and not env_only:
        print("\n-- Data files (raw) --")
        for key in ["dukascopy_file", "forexsb_file", "fred_file"]:
            path = cfg["data"].get(key)
            check(f"Raw data file present: {path}", lambda p=path: bool(p) and os.path.isfile(p))

        print("\n-- Data files (processed) --")
        for split in ("train", "val", "test"):
            for prefix in ("X", "y"):
                path = os.path.join(cfg["data"]["processed_dir"], f"{prefix}_{split}.npy")
                check(f"Processed file present: {path}", lambda p=path: os.path.isfile(p))

    print("\n-- API keys --")
    if env_only:
        check("NVIDIA_API_KEY (advisory; needed only for LLM stages)",
              lambda: "set" if os.environ.get("NVIDIA_API_KEY") else "not set -- set it before running the LLM stages")
    else:
        check("NVIDIA_API_KEY environment variable set",
              lambda: bool(os.environ.get("NVIDIA_API_KEY")))
    check("FRED_API_KEY set (optional; without it acquisition uses FRED's keyless CSV endpoint)",
          lambda: "set" if os.environ.get("FRED_API_KEY") else "not set -- OK, keyless fallback will be used")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")
    print("=" * 70)
    if CHECKS_FAILED:
        print("\nFailed checks (see RUNNING.md / SETUP.md for how to resolve each):")
        for c in CHECKS_FAILED:
            print(f"  - {c}")
        sys.exit(1)
    else:
        print("\nEnvironment looks ready.")


if __name__ == "__main__":
    main()
