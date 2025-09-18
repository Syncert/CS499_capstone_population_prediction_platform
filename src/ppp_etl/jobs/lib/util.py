import os, json, datetime as dt
from pathlib import Path

def batch_id(prefix="week2"):
    return f"{prefix}_{dt.date.today().isoformat()}"

def artifacts_dir() -> Path:
    p = Path("artifacts/validation")
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))