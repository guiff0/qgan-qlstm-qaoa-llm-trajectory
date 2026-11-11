"""Read a models/*.ckpt WITHOUT torch (pure numpy + pickle), so the inventory works on any machine and in the unit tests.

A torch checkpoint is a zip: `<name>/data.pkl` (a pickle whose tensors are persistent ids) and `<name>/data/<key>` (raw bytes).
This reader rebuilds the tensors as numpy arrays and refuses to execute anything else: find_class only allows the few torch
helpers a state_dict needs plus collections.OrderedDict; any other global raises. Read-only; never writes."""
from __future__ import annotations

import collections
import io
import os
import pickle
import zipfile

import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
       "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8,
       "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, arr):
        self.arr = arr


def _rebuild_tensor(storage, offset, size, stride, *_a, **_k):
    base = storage.arr
    if not size:
        return base[offset:offset + 1].reshape(()).copy()
    return np.lib.stride_tricks.as_strided(base[offset:], shape=tuple(size),
                                           strides=tuple(s * base.itemsize for s in stride)).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_a, **_k: data
        if module == "torch" and name in _DT:
            return name
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name} from a checkpoint")

    def persistent_load(self, pid):
        _kind, stype, key, _loc, numel = pid[:5]
        dt = _DT[stype if isinstance(stype, str) else getattr(stype, "__name__", str(stype))]
        return _Storage(np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=dt, count=int(numel)))


def load_ckpt(path: str) -> dict:
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    return _Unpickler(io.BytesIO(zf.read(pkl)), zf, pkl.rsplit("/", 1)[0]).load()


def pid_alive(pid: int) -> bool:
    """Safe on Windows: os.kill(pid, 0) there TERMINATES the process, so it is never used on nt."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def describe(path: str) -> dict:
    """One checkpoint -> a flat, JSON-able record. Never raises: an unreadable file is reported as such."""
    name = os.path.basename(path)
    stem = name[:-len(".ckpt")]
    prefix, _, fp_name = stem.rpartition("_")
    rec = {"file": name, "prefix": prefix, "fingerprint_in_name": fp_name, "bytes": os.path.getsize(path),
           "mtime_utc": _utc(os.path.getmtime(path))}
    try:
        c = load_ckpt(path)
    except Exception as exc:  # noqa: BLE001
        return {**rec, "state": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"}
    hist = list(c.get("entanglement_history") or [])
    gen = c.get("generator") or {}
    theta = next((v for k, v in gen.items() if k.split(".")[-1] == "theta"), None)
    n_gen = int(sum(np.asarray(v).size for v in gen.values()))
    head = c.get("forecast_head") or {}
    lin = next((np.asarray(v) for k, v in head.items() if k.endswith("linear.weight") or k == "weight"), None)
    epoch = int(c.get("next_epoch", 0))
    rec.update({
        "fingerprint": c.get("fingerprint"), "fingerprint_matches_name": c.get("fingerprint") == fp_name,
        "completed": bool(c.get("completed")), "next_epoch": epoch, "target_frame": c.get("target_frame"),
        "state": "COMPLETE" if c.get("completed") else f"INCOMPLETE (next epoch {epoch})",
        "head_mse": _f(c.get("head_mse")), "head_fitted": not _isnan(c.get("head_mse")),
        "head_input_dim": int(lin.shape[-1]) if lin is not None else None,
        "generator_params": n_gen, "theta_params": int(np.asarray(theta).size) if theta is not None else None,
        "theta_sd": float(np.asarray(theta).std()) if theta is not None else None,
        "entropy_points": len(hist),
        "entropy_first": _f(hist[0].get("entanglement_entropy")) if hist else None,
        "entropy_last": _f(hist[-1].get("entanglement_entropy")) if hist else None,
        "entropy_max_possible": _f(hist[-1].get("max_possible_entropy")) if hist else None,
        "last_entropy_epoch": hist[-1].get("epoch") if hist else None,
        "has_optimizer_state": bool(c.get("g_optimizer")) and bool(c.get("d_optimizer")),
    })
    return rec


def lock_info(ckpt_path: str) -> dict | None:
    """A training lock next to the checkpoint: `<ckpt>.train.lock` (also accepted: `<stem>_ckpt_train.lock`, the name a
    file gets when the dots are replaced, e.g. after an upload). Holds 'pid\\nUTC timestamp'."""
    stem = ckpt_path[:-len(".ckpt")]
    for cand in (ckpt_path + ".train.lock", stem + "_ckpt_train.lock"):
        if os.path.isfile(cand):
            try:
                parts = open(cand, encoding="utf8").read().split()
                pid = int(parts[0])
            except Exception:  # noqa: BLE001
                return {"lock_file": os.path.basename(cand), "pid": None, "utc": None, "alive": False}
            return {"lock_file": os.path.basename(cand), "pid": pid, "utc": parts[1] if len(parts) > 1 else None,
                    "alive": pid_alive(pid)}
    return None


def synth_cache(ckpt_path: str) -> dict | None:
    p = ckpt_path[:-len(".ckpt")] + "_synth.npy"
    if not os.path.isfile(p):
        return None
    try:
        a = np.load(p, mmap_mode="r")
        return {"file": os.path.basename(p), "rows": int(a.shape[0]), "dims": int(a.shape[1]) if a.ndim > 1 else 1}
    except Exception:  # noqa: BLE001
        return {"file": os.path.basename(p), "rows": None, "dims": None}


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _isnan(x) -> bool:
    return _f(x) is None


def _utc(ts: float) -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
