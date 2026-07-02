"""Minimal env loader: populate os.environ from a .env file if present."""
import os

def load_env(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else [
        os.path.join(here, os.pardir, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return True
    return False
