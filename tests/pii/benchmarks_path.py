import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "benchmarks" / "pii_eval.py"
_spec = importlib.util.spec_from_file_location("pii_eval", _p)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate = _mod.evaluate
