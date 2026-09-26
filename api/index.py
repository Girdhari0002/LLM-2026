"""Vercel entrypoint for the FastAPI application."""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_SOURCE = PROJECT_ROOT / "python-api" / "main.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("llm_api_main", API_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load FastAPI application from {API_SOURCE}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.app
