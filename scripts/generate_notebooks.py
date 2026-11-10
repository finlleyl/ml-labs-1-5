"""Создает ноутбуки с одинаковой схемой эксперимента."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

LABS = {
    1: {
        "slug": "knn",
        "title": "KNN",
        "principle": "KNN смотрит на ближайшие объекты из train. Для класса берется голосование соседей, для числового ответа их среднее. Поэтому масштаб признаков и выбор расстояния заметно влияют на результат.",
        "hypotheses": "Проверяю стандартизацию и удаление константных признаков. Заодно подбираю число соседей, тип весов и норму расстояния.",
        "custom": "Своя версия считает расстояния через NumPy. Прогноз идет небольшими блоками, чтобы не хранить в памяти лишние матрицы.",
    },
    2: {
        "slug": "regression",
        "title": "Логистическая и линейная регрессия",
        "principle": "Логистическая регрессия возвращает вероятность класса через сигмоиду. Линейная регрессия минимизирует квадраты ошибок, а полиномиальные признаки позволяют учесть нелинейную связь состава бетона и прочности.",
        "hypotheses": "Для классификации проверяю масштабирование, L2-регуляризацию и балансировку классов. Для бетона сравниваю исходные признаки с полиномом второй степени.",
        "custom": "Логистическая регрессия обучается градиентным спуском. Для линейной используется `np.linalg.lstsq`: так устойчивее, чем явное обращение матрицы.",
    },
    3: {
        "slug": "decision_tree",
        "title": "Решающее дерево",
        "principle": "Дерево делит данные порогами признаков. В классификации используется индекс Джини, в регрессии дисперсия ответа. Без ограничений дерево легко подгоняется под train.",
        "hypotheses": "Подбираю глубину и минимальный размер листа. Масштабирование здесь не нужно: дерево сравнивает признаки с порогами, а не считает расстояния.",
        "custom": "В своей версии пороги ищутся по отсортированным значениям. Кумулятивные суммы ускоряют расчет качества разбиений.",
    },
    4: {
        "slug": "random_forest",
        "title": "Случайный лес",
        "principle": "Лес усредняет деревья, обученные на разных бутстрэп-выборках. Случайный набор признаков в узлах делает деревья менее похожими друг на друга.",
        "hypotheses": "Меняю число деревьев, глубину и размер листа. Для классификации беру `sqrt` признаков в узле, для регрессии 80%. В своей версии оставляю не больше 50 деревьев ради времени расчета.",
        "custom": "Лес собран из собственных деревьев из третьей работы. Каждое дерево получает свою выборку и свой seed.",
    },
    5: {
        "slug": "gradient_boosting",
        "title": "Градиентный бустинг",
        "principle": "Бустинг добавляет деревья последовательно. Каждое следующее исправляет ошибку текущего ансамбля: остаток в регрессии или градиент логистической потери в классификации.",
        "hypotheses": "Подбираю скорость обучения, число деревьев и глубину. Маленький шаг обычно требует большего числа деревьев, поэтому эти параметры рассматриваются вместе.",
        "custom": "В своей версии базовые модели являются регрессионными деревьями. Их прогнозы складываются с коэффициентом `learning_rate`.",
    },
}

IMPORTS = '''# Подключаю общие функции и настраиваю вывод таблиц.
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.experiments import TaskExperiment

pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
plt.style.use("seaborn-v0_8-whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)
print(f"Корень проекта: {ROOT}")'''


def add_pair(nb, markdown, code):
    nb.cells.append(nbf.v4.new_markdown_cell(markdown))
    nb.cells.append(nbf.v4.new_code_cell(code))


def build_lab(number, spec):
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}})
    nb.cells.append(nbf.v4.new_markdown_cell(f"""# Лабораторная работа №{number}. {spec['title']}

Чибугаев Илья Алексеевич, вариант 22.

## Данные и метрики

Для классификации взят [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere): 351 радиосигнал и 34 признака. Класс `bad=1` означает сигнал без структуры в ионосфере. Это задача автоматической проверки радиосигналов.

Для регрессии взят [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength): 1030 составов бетона и 8 признаков. По составу и возрасту нужно оценить прочность в МПа. Такой прогноз полезен перед лабораторными испытаниями смеси.

{spec['principle']}

Train и test соотносятся как 80/20, `random_state=22`. Классы при разбиении сохраняются в одинаковой пропорции. Подбор параметров идет только на train.

В классификации основной показатель: ROC-AUC. Он не зависит от выбранного порога. F1 показывает качество для `bad`, accuracy дает общую долю верных ответов. В регрессии основной показатель: RMSE в МПа. MAE проще интерпретировать как типичную ошибку, а R² показывает долю объясненной дисперсии.
"""))
    add_pair(nb, "## Подготовка\n\nОбщий код вынесен в `src`, чтобы не повторять его в каждой работе.", IMPORTS)
    add_pair(nb, "## Проверка данных\n\nСмотрю размеры выборок, пропуски и целевые переменные. Эти разбиения используются во всех работах.", f'''# Загружаю данные и проверяю размеры выборок.
experiments = {{task: TaskExperiment({number}, task) for task in ("classification", "regression")}}
for task, experiment in experiments.items():
    data = experiment.data
    print(f"{{task}}: {{data.name}}")
    print(data.practical_task)
    print(f"train={{data.X_train.shape}}, test={{data.X_test.shape}}, пропуски={{int(data.X.isna().sum().sum())}}")
    if task == "classification":
        print(f"Доля bad: train={{data.y_train.mean():.3f}}, test={{data.y_test.mean():.3f}}")
    else:
        print(data.y.describe().to_string())
    print()''')
    if number == 1:
        add_pair(nb, "## Взгляд на данные\n\nПроверяю баланс классов и константные сигналы. Для бетона смотрю распределение прочности и простые корреляции с признаками.", '''# Строю распределения и считаю корреляции для бетона.
classification = experiments["classification"].data
regression = experiments["regression"].data
constant_features = classification.X_train.columns[classification.X_train.nunique() <= 1].tolist()
print("Константные признаки Ionosphere:", constant_features)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
classification.y_train.map({0: "good", 1: "bad"}).value_counts().plot(kind="bar", ax=axes[0], color=["#3178c6", "#d1495b"])
axes[0].set_title("Ionosphere: классы в train")
axes[0].set_ylabel("Число объектов")
axes[0].tick_params(axis="x", rotation=0)
regression.y_train.hist(bins=25, ax=axes[1], color="#2a9d8f")
axes[1].set_title("Concrete: прочность в train")
axes[1].set_xlabel("МПа")
plt.tight_layout()
plt.show()

corr = regression.X_train.assign(strength_mpa=regression.y_train).corr()["strength_mpa"].drop("strength_mpa").sort_values()
display(corr.to_frame("Корреляция с прочностью"))''')
    add_pair(nb, f"## {'Бейзлайн' if number != 1 else 'Базовый бейзлайн'} sklearn\n\nБеру модель с параметрами по умолчанию. Медианный imputer оставлен как страховка, хотя пропусков в данных нет.", '''# Обучаю базовые модели sklearn.
for experiment in experiments.values():
    experiment.fit_sklearn_baseline()
baseline_results = pd.concat([experiment.results for experiment in experiments.values()], ignore_index=True)
display(baseline_results.drop(columns=["params"]))''')
    add_pair(nb, f"## Подбор параметров\n\n{spec['hypotheses']}\n\nИспользую пятикратную CV. Для классификации выбираю по ROC-AUC, для регрессии по RMSE.", '''# Подбираю параметры только на обучающей выборке.
for experiment in experiments.values():
    experiment.fit_sklearn_improved()
    print(f"{{experiment.task}}: {{experiment.search.best_params_}}")
    display(experiment.cv_table().head())''')
    add_pair(nb, "## Что дала настройка\n\nСравниваю метрики до и после. Для ROC-AUC, F1, accuracy и R² рост означает улучшение, для RMSE и MAE наоборот.", '''# Считаю разницу метрик между базовой и настроенной моделями.
metric_names = {"classification": ["roc_auc", "f1", "accuracy"], "regression": ["rmse", "mae", "r2"]}
for task, experiment in experiments.items():
    rows = experiment.results.set_index("stage")
    metrics = metric_names[task]
    comparison = rows.loc[["baseline", "improved"], metrics].T
    comparison["change"] = comparison["improved"] - comparison["baseline"]
    print(task)
    display(comparison)''')
    add_pair(nb, f"## Своя реализация\n\n{spec['custom']}\n\nКод лежит в `src/models/`. Сначала обучаю базовую версию на тех же данных, что и sklearn.", '''# Обучаю базовые версии собственных моделей.
for experiment in experiments.values():
    experiment.fit_custom_baseline()
custom_baseline_results = pd.concat([
    experiment.results[experiment.results["implementation"] == "from_scratch"]
    for experiment in experiments.values()
], ignore_index=True)
display(custom_baseline_results.drop(columns=["params"]))''')
    add_pair(nb, "## Своя модель после настройки\n\nПереношу препроцессинг и параметры, выбранные на CV. Ограничение по числу деревьев для своего леса указано выше.", '''# Применяю найденные параметры к своим моделям.
for experiment in experiments.values():
    experiment.fit_custom_improved()
all_lab_results = pd.concat([experiment.results for experiment in experiments.values()], ignore_index=True)
display(all_lab_results.drop(columns=["params"]))''')
    add_pair(nb, "## Результаты\n\nСохраняю все четыре варианта для каждой задачи. Вывод зависит от полученных чисел: если настройка не помогла, это будет видно.", f'''# Сохраняю таблицу и вывожу изменение основной метрики.
output_path = ROOT / "results" / "lab{number:02d}_results.csv"
output_path.parent.mkdir(exist_ok=True)
all_lab_results.to_csv(output_path, index=False)

def verdict(change, higher_is_better):
    if np.isclose(change, 0):
        return "не изменился"
    improved = change > 0 if higher_is_better else change < 0
    return "улучшился" if improved else "ухудшился"

for task, experiment in experiments.items():
    frame = experiment.results.set_index(["implementation", "stage"])
    metrics = metric_names[task]
    primary = "roc_auc" if task == "classification" else "rmse"
    higher_is_better = primary != "rmse"
    sklearn_base = frame.loc[("sklearn", "baseline"), primary]
    sklearn_best = frame.loc[("sklearn", "improved"), primary]
    custom_base = frame.loc[("from_scratch", "baseline"), primary]
    custom_best = frame.loc[("from_scratch", "improved"), primary]
    print(f"{{task}}, sklearn: {{primary}} {{sklearn_base:.4f}} → {{sklearn_best:.4f}}, "
          f"{{verdict(sklearn_best - sklearn_base, higher_is_better)}}.")
    print(f"{{task}}, своя версия: {{primary}} {{custom_base:.4f}} → {{custom_best:.4f}}, "
          f"{{verdict(custom_best - custom_base, higher_is_better)}}.")
    print(f"Параметры sklearn: {{experiment.search.best_params_}}")
print(f"Файл: {{output_path.relative_to(ROOT)}}")''')
    path = OUT / f"lab{number:02d}_{spec['slug']}.ipynb"
    nbf.write(nb, path)
    return path


def build_comparison():
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}})
    nb.cells.append(nbf.v4.new_markdown_cell("""# Сравнение моделей из лабораторных 1–5

Чибугаев Илья Алексеевич, вариант 22.

Здесь собраны результаты всех работ. Классификация и регрессия сравниваются отдельно: ROC-AUC и RMSE нельзя ставить в одну шкалу. Для выводов берутся настроенные модели, а бейзлайны нужны для проверки эффекта настройки.
"""))
    add_pair(nb, "## Загрузка результатов\n\nПроверяю, что есть все пять файлов и нет повторов. Всего должно получиться 40 строк.", '''# Читаю CSV и проверяю полноту результатов.
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
plt.style.use("seaborn-v0_8-whitegrid")
paths = sorted((ROOT / "results").glob("lab??_results.csv"))
assert len(paths) == 5, f"Нужно 5 файлов, найдено: {len(paths)}"
results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
assert len(results) == 40, f"Нужно 40 строк, получено: {len(results)}"
assert set(results["lab"]) == {1, 2, 3, 4, 5}
assert not results.duplicated(["lab", "task", "implementation", "stage"]).any()
results.to_csv(ROOT / "results" / "all_results.csv", index=False)
print(f"Строк: {len(results)}, файлов: {len(paths)}")
display(results.drop(columns=["params"]))''')
    add_pair(nb, "## До и после настройки\n\nСвожу основную метрику в одну таблицу. Отрицательное изменение RMSE говорит об улучшении, для ROC-AUC нужен положительный знак.", '''# Собираю основную метрику по каждому алгоритму.
summary_rows = []
for (lab, algorithm, task, implementation), group in results.groupby(["lab", "algorithm", "task", "implementation"]):
    indexed = group.set_index("stage")
    metric = "roc_auc" if task == "classification" else "rmse"
    summary_rows.append({
        "lab": lab, "algorithm": algorithm, "task": task, "implementation": implementation,
        "metric": metric, "baseline": indexed.loc["baseline", metric],
        "improved": indexed.loc["improved", metric],
        "change": indexed.loc["improved", metric] - indexed.loc["baseline", metric],
    })
summary = pd.DataFrame(summary_rows).sort_values(["task", "implementation", "lab"])
summary.to_csv(ROOT / "results" / "summary_metrics.csv", index=False)
display(summary)''')
    add_pair(nb, "## Графики\n\nПоказываю настроенные версии. F1 и MAE добавлены, чтобы видеть качество не только по основной метрике.", '''# Строю графики для классификации и регрессии.
for task in ("classification", "regression"):
    subset = results[(results["task"] == task) & (results["stage"] == "improved")].copy()
    subset["model"] = subset["algorithm"] + "\\n" + subset["implementation"]
    metrics = ["roc_auc", "f1"] if task == "classification" else ["rmse", "mae"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    for axis, metric in zip(axes, metrics):
        colors = ["#3178c6" if value == "sklearn" else "#e76f51" for value in subset["implementation"]]
        axis.bar(subset["model"], subset[metric], color=colors)
        axis.set_title(f"{task}: {metric}")
        axis.tick_params(axis="x", rotation=65)
        axis.bar_label(axis.containers[0], fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    figure_path = ROOT / "results" / f"comparison_{task}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Файл: {figure_path.relative_to(ROOT)}")''')
    add_pair(nb, "## Вывод\n\nВыбираю лидеров по ROC-AUC и RMSE. Близкие результаты на одном test не дают оснований говорить о статистически значимом преимуществе.", '''# Нахожу лучшие модели и вывожу итоговые значения.
conclusions = []
for task in ("classification", "regression"):
    metric = "roc_auc" if task == "classification" else "rmse"
    improved = results[(results["task"] == task) & (results["stage"] == "improved")]
    improved = improved.sort_values(metric, ascending=metric == "rmse")
    best = improved.iloc[0]
    best_sklearn = improved[improved["implementation"] == "sklearn"].iloc[0]
    best_custom = improved[improved["implementation"] == "from_scratch"].iloc[0]
    conclusions.extend([
        f"### {task}",
        f"Лучший результат: {best['algorithm']} ({best['implementation']}), {metric} = {best[metric]:.4f}.",
        f"Из sklearn лидирует {best_sklearn['algorithm']}: {best_sklearn[metric]:.4f}.",
        f"Из собственных моделей лидирует {best_custom['algorithm']}: {best_custom[metric]:.4f}.",
    ])
conclusions.extend([
    "### Ограничение",
    "При выборе модели стоит учитывать разброс на CV и цену ошибки. Один test не доказывает преимущество при маленькой разнице метрик.",
])
display(Markdown("\\n".join(conclusions)))''')
    path = OUT / "06_comparison.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    generated = [build_lab(number, spec) for number, spec in LABS.items()]
    generated.append(build_comparison())
    for path in generated:
        print(path.relative_to(ROOT))
