"""Проверка структуры notebook-файлов и полноты сохранённых результатов."""
from pathlib import Path
import ast
import re
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors = []
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    expected_minimum = 4 if path.name == "06_comparison.ipynb" else 8
    if len(code_cells) < expected_minimum:
        errors.append(f"{path.name}: только {len(code_cells)} кодовых ячеек")
    for number, cell in enumerate(code_cells, start=1):
        try:
            ast.parse(cell.source)
        except SyntaxError as error:
            errors.append(f"{path.name}, ячейка {number}: {error}")
        first_line = next((line.strip() for line in cell.source.splitlines() if line.strip()), "")
        if not first_line.startswith("#") or not re.search(r"[А-Яа-яЁё]", first_line):
            errors.append(f"{path.name}, ячейка {number}: нет русского пояснения в начале")
        for line in cell.source.splitlines():
            if line.lstrip().startswith("#") and not re.search(r"[А-Яа-яЁё]", line):
                errors.append(f"{path.name}, ячейка {number}: комментарий не на русском: {line.strip()}")
        if cell.execution_count is None:
            errors.append(f"{path.name}, ячейка {number}: не выполнена")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(f"{path.name}, ячейка {number}: ошибка {output.ename}")
        if re.search(r"\b(TODO|FIXME|PLACEHOLDER)\b", cell.source, re.IGNORECASE):
            errors.append(f"{path.name}, ячейка {number}: найден незаполненный маркер")
    print(f"{path.name}: {len(notebook.cells)} ячеек всего, {len(code_cells)} с кодом")

results = pd.read_csv(ROOT / "results" / "all_results.csv")
if len(results) != 40:
    errors.append(f"В all_results.csv {len(results)} строк вместо 40")
if results.duplicated(["lab", "task", "implementation", "stage"]).any():
    errors.append("В результатах есть повторяющиеся конфигурации")
for task, metric in (("classification", "roc_auc"), ("regression", "rmse")):
    subset = results[results["task"] == task]
    if len(subset) != 20 or subset[metric].isna().any():
        errors.append(f"Неполные результаты {task}/{metric}")

if errors:
    raise SystemExit("\n".join(errors))
print("Проверка пройдена: 6 выполненных notebook-файлов, русские комментарии, 40 результатов.")
