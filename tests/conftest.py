import os
from pathlib import Path


# Évite les répertoires temporaires utilisateur parfois non accessibles sous Windows.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".tmp" / "matplotlib"))
