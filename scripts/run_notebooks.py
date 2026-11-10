"""Выполняет пять работ, затем итоговое сравнение."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("lab??_*.ipynb"))
NOTEBOOKS.append(ROOT / "notebooks" / "06_comparison.ipynb")

for notebook in NOTEBOOKS:
    print(f"\n=== {notebook.name} ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", str(notebook)],
        cwd=ROOT,
        check=True,
    )
print(f"Готово: {len(NOTEBOOKS)} ноутбуков.")
