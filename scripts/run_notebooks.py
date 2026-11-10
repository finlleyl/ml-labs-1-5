"""Execute every notebook in the required order from the repository root."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
for notebook in NOTEBOOKS:
    print(f"\n=== Executing {notebook.name} ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", str(notebook)],
        cwd=ROOT,
        check=True,
    )
print(f"Executed {len(NOTEBOOKS)} notebooks successfully.")
