"""Generate readable, fully commented laboratory notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

LABS = {
    1: {
        "slug": "knn", "title": "Проведение исследований с алгоритмом KNN",
        "principle": "KNN сохраняет обучающие объекты и относит новый объект к классу большинства либо усредняет ответы ближайших соседей. Поэтому масштаб признаков, метрика расстояния и число соседей непосредственно определяют качество.",
        "hypotheses": [
            "Стандартизация признаков устранит доминирование переменных с большим масштабом в расстоянии Минковского.",
            "Удаление константных признаков не потеряет информацию, но сократит вычисления.",
            "Число соседей, взвешивание по обратному расстоянию и норма p будут подобраны по 5-блочной кросс-валидации.",
        ],
        "custom": "Расстояния считаются матричными операциями NumPy; для экономии памяти прогноз строится блоками. Взвешивание обрабатывает нулевые расстояния без деления на ноль.",
    },
    2: {
        "slug": "regression", "title": "Проведение исследований с логистической и линейной регрессией",
        "principle": "Логистическая регрессия оценивает вероятность класса через сигмоиду линейной комбинации признаков. Линейная регрессия минимизирует сумму квадратов ошибок; полиномиальные признаки позволяют ей описывать нелинейность в исходных переменных.",
        "hypotheses": [
            "Стандартизация и регуляризация улучшат устойчивость логистической регрессии при коррелированных сигналах.",
            "Балансировка классов проверяется как способ повысить качество распознавания редкого класса bad, но выбор делается по ROC-AUC на CV.",
            "Для бетона проверяются попарные взаимодействия и квадраты признаков: зависимость прочности от состава и возраста нелинейна.",
        ],
        "custom": "Логистическая регрессия обучается полным градиентным спуском со стабильной сигмоидой и L2-регуляризацией. Линейная регрессия использует np.linalg.lstsq вместо явного обращения матрицы, что устойчивее при мультиколлинеарности.",
    },
    3: {
        "slug": "decision_tree", "title": "Проведение исследований с решающим деревом",
        "principle": "CART рекурсивно делит пространство признаков, минимизируя неоднородность листьев: индекс Джини для классификации и дисперсию ответа для регрессии. Неограниченное дерево склонно запоминать обучающую выборку.",
        "hypotheses": [
            "Ограничение глубины и минимального числа объектов в листе должно уменьшить переобучение.",
            "Масштабирование не требуется: пороговые разбиения инвариантны к монотонному преобразованию каждого признака.",
            "Оптимальные max_depth и min_samples_leaf выбираются только по обучающей части на 5-блочной CV.",
        ],
        "custom": "На каждом узле перебираются допустимые пороги отсортированных признаков. Кумулятивные суммы дают вычисление Джини и SSE без повторного прохода по каждому кандидату.",
    },
    4: {
        "slug": "random_forest", "title": "Проведение исследований со случайным лесом",
        "principle": "Случайный лес усредняет деревья, обученные на бутстрэп-выборках и случайных подмножествах признаков. Это снижает дисперсию отдельного дерева за счёт декоррелирования ошибок.",
        "hypotheses": [
            "Ограничение глубины и размера листа снизит переобучение, а увеличение числа деревьев стабилизирует усреднение.",
            "Случайный подбор признаков используется для классификации (sqrt), а для регрессии проверяется доля 0.8 признаков.",
            "В собственной реализации используется до 50 деревьев, чтобы чистый NumPy/Python-код оставался воспроизводимым за разумное время; остальные выбранные параметры сохраняются.",
        ],
        "custom": "Каждое дерево — собственная CART-реализация из лабораторной 3. Бутстрэп и случайные подмножества признаков создаются независимыми генераторами с фиксированными seed.",
    },
    5: {
        "slug": "gradient_boosting", "title": "Проведение исследований с градиентным бустингом",
        "principle": "Градиентный бустинг последовательно добавляет неглубокие регрессионные деревья, обученные на антиградиент функции потерь. Для классификации используется логистическая потеря, для регрессии — квадратичная.",
        "hypotheses": [
            "Уменьшение learning_rate с увеличением числа деревьев может улучшить обобщение за счёт более мелких шагов.",
            "Неглубокие деревья (глубина 2–3) ограничивают сложность базовых моделей и взаимодействия признаков.",
            "Число деревьев, скорость обучения и глубина подбираются по 5-блочной CV на обучающей выборке.",
        ],
        "custom": "Базовые модели — собственные регрессионные деревья. В классификации деревья аппроксимируют y−sigmoid(F), в регрессии — остатки y−F; итоговый прогноз является взвешенной суммой деревьев.",
    },
}

IMPORTS = '''# Настраиваем воспроизводимое окружение и импортируем общий код эксперимента.
from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.experiments import TaskExperiment, LAB_NAMES

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

**Студент:** Чибугаев Илья Алексеевич  
**Вариант:** 22  
**Случайное зерно:** 22

## Исходные данные

- **Классификация:** [Ionosphere, UCI](https://archive.ics.uci.edu/dataset/52/ionosphere) — 351 объект, 34 числовых признака; `bad=1` обозначает сигнал без структуры в ионосфере.
- **Регрессия:** [Concrete Compressive Strength, UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) — 1030 составов бетона, 8 признаков, целевая переменная — прочность в МПа.

Датасеты выбраны для разных практических задач: контроль качества радиосигнала и предварительная оценка рецептуры бетона. Оба набора небольшие, содержат только числовые признаки и позволяют сопоставить алгоритмы без изменения задачи между лабораторными.

{spec['principle']}

## Протокол и метрики

Для обеих задач используется фиксированное разбиение **80% / 20%** с `random_state=22`; в классификации разбиение стратифицировано. Подбор параметров выполняется только внутри обучающей выборки. Все четыре конфигурации (sklearn / собственная, базовая / улучшенная) оцениваются на одном и том же тестовом наборе.

- **Классификация:** основная метрика — **ROC-AUC**, поскольку она оценивает ранжирование и не зависит от одного порога; **F1** контролирует качество по классу `bad`, а **accuracy** показывает общую долю верных ответов.
- **Регрессия:** основная метрика — **RMSE (МПа)**, штрафующая крупные ошибки; **MAE (МПа)** даёт типичную абсолютную ошибку, **R²** — долю объяснённой дисперсии.
"""))
    add_pair(nb, "## 1. Подготовка\n\nКод импортирует только общие функции и фиксирует настройки отображения. Каждая следующая ячейка имеет отдельный комментарий.", IMPORTS)
    add_pair(nb, "### Загрузка и проверка разбиений\n\nПроверяем размеры, отсутствие пропусков, баланс классов и диапазон целевой переменной. Разбиения одинаковы во всех пяти работах.", f'''# Создаём эксперименты для двух задач и выводим контрольные характеристики данных.
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
        add_pair(nb, "## 2. Исследование данных\n\nВизуализация проверяет исходные предположения: дисбаланс классов, константные сигналы и нелинейную связь состава бетона с прочностью. Графики не используются для подбора на тестовой выборке.", '''# Строим распределение классов, гистограмму прочности и корреляции признаков бетона.
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
display(corr.to_frame("Корреляция Пирсона с прочностью"))''')
    add_pair(nb, f"## {'3' if number == 1 else '2'}. Базовый бейзлайн sklearn\n\nБейзлайн использует параметры алгоритма по умолчанию и только медианное заполнение пропусков. Это точка отсчёта, а не ожидаемый победитель.", f'''# Обучаем базовые sklearn-модели и оцениваем их на отложенной выборке.
for experiment in experiments.values():
    experiment.fit_sklearn_baseline()
baseline_results = pd.concat([experiment.results for experiment in experiments.values()], ignore_index=True)
display(baseline_results.drop(columns=["params"]))''')
    hypotheses = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(spec["hypotheses"]))
    add_pair(nb, f"## {'4' if number == 1 else '3'}. Улучшение бейзлайна\n\n### Проверяемые гипотезы\n\n{hypotheses}\n\nМетрика подбора — ROC-AUC для классификации и RMSE для регрессии. Лучшие параметры затем один раз оцениваются на тесте.", f'''# Запускаем GridSearchCV и показываем пять лучших конфигураций для каждой задачи.
for experiment in experiments.values():
    experiment.fit_sklearn_improved()
    print(f"{{experiment.task}}: лучшие параметры = {{experiment.search.best_params_}}")
    display(experiment.cv_table().head())''')
    add_pair(nb, "### Сравнение базового и улучшенного sklearn-бейзлайна\n\nПоложительная разница ROC-AUC/F1/accuracy/R² означает улучшение; для RMSE/MAE улучшение соответствует отрицательной разнице.", '''# Формируем таблицу изменений метрик между улучшенной и базовой sklearn-моделями.
metric_names = {"classification": ["roc_auc", "f1", "accuracy"], "regression": ["rmse", "mae", "r2"]}
for task, experiment in experiments.items():
    rows = experiment.results.set_index("stage")
    metrics = metric_names[task]
    comparison = rows.loc[["baseline", "improved"], metrics].T
    comparison["change"] = comparison["improved"] - comparison["baseline"]
    print(task)
    display(comparison)''')
    add_pair(nb, f"## {'5' if number == 1 else '4'}. Собственная реализация\n\n{spec['custom']}\n\nИсходный код находится в `src/models/`. Сначала используются базовые параметры, затем техники и параметры улучшенного бейзлайна.", '''# Обучаем собственные базовые модели на тех же преобразованных данных, что и sklearn-бейзлайн.
for experiment in experiments.values():
    experiment.fit_custom_baseline()
custom_baseline_results = pd.concat([
    experiment.results[experiment.results["implementation"] == "from_scratch"]
    for experiment in experiments.values()
], ignore_index=True)
display(custom_baseline_results.drop(columns=["params"]))''')
    add_pair(nb, "### Собственная модель с улучшенным бейзлайном\n\nПреобразования обучены только на train. Для собственной модели переносятся выбранные на CV параметры; исключение по числу деревьев случайного леса явно указано в гипотезах.", '''# Добавляем улучшенный препроцессинг и выбранные параметры в собственные реализации.
for experiment in experiments.values():
    experiment.fit_custom_improved()
all_lab_results = pd.concat([experiment.results for experiment in experiments.values()], ignore_index=True)
display(all_lab_results.drop(columns=["params"]))''')
    add_pair(nb, "## 6. Сохранение результатов и выводы\n\nВывод формируется по фактическим значениям, а не по предположению, что улучшение обязано помочь. На маленькой тестовой выборке небольшие различия не следует трактовать как статистически доказанное превосходство.", f'''# Сохраняем четыре результата на задачу и печатаем аргументированное заключение по каждой задаче.
output_path = ROOT / "results" / "lab{number:02d}_results.csv"
output_path.parent.mkdir(exist_ok=True)
all_lab_results.to_csv(output_path, index=False)

for task, experiment in experiments.items():
    frame = experiment.results.set_index(["implementation", "stage"])
    metrics = metric_names[task]
    sklearn_base = frame.loc[("sklearn", "baseline"), metrics]
    sklearn_best = frame.loc[("sklearn", "improved"), metrics]
    custom_base = frame.loc[("from_scratch", "baseline"), metrics]
    custom_best = frame.loc[("from_scratch", "improved"), metrics]
    primary = "roc_auc" if task == "classification" else "rmse"
    higher_is_better = primary != "rmse"
    sklearn_change = sklearn_best[primary] - sklearn_base[primary]
    custom_change = custom_best[primary] - custom_base[primary]
    sklearn_verdict = "улучшился" if (sklearn_change > 0) == higher_is_better else "не улучшился"
    custom_verdict = "улучшился" if (custom_change > 0) == higher_is_better else "не улучшился"
    print(f"{{task}}: sklearn {{primary}} {{sklearn_base[primary]:.4f}} → {{sklearn_best[primary]:.4f}} ({{sklearn_verdict}}); "
          f"своя реализация {{custom_base[primary]:.4f}} → {{custom_best[primary]:.4f}} ({{custom_verdict}}).")
    print(f"Лучшие параметры sklearn: {{experiment.search.best_params_}}")
print(f"Результаты сохранены: {{output_path.relative_to(ROOT)}}")''')
    path = OUT / f"lab{number:02d}_{spec['slug']}.ipynb"
    nbf.write(nb, path)
    return path


def build_comparison():
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}})
    nb.cells.append(nbf.v4.new_markdown_cell("""# Итоговое сравнение алгоритмов лабораторных 1–5

**Студент:** Чибугаев Илья Алексеевич, вариант 22.

Notebook объединяет результаты пяти работ на неизменных train/test-разбиениях. Сравнение проводится отдельно для классификации и регрессии, поскольку их метрики и шкалы несопоставимы. Основной практический вывод делается по улучшенным моделям; бейзлайны сохранены для оценки эффекта настройки.
"""))
    add_pair(nb, "## 1. Загрузка и контроль полноты\n\nОжидаются пять файлов, в каждом — четыре конфигурации для двух задач (восемь строк). Отсутствие файла или строки останавливает анализ.", '''# Читаем результаты всех лабораторных и проверяем полноту перед сравнением.
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
plt.style.use("seaborn-v0_8-whitegrid")
paths = sorted((ROOT / "results").glob("lab??_results.csv"))
assert len(paths) == 5, f"Ожидалось 5 файлов, найдено: {len(paths)}"
results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
assert len(results) == 40, f"Ожидалось 40 строк, получено: {len(results)}"
assert set(results["lab"]) == {1, 2, 3, 4, 5}
assert not results.duplicated(["lab", "task", "implementation", "stage"]).any()
results.to_csv(ROOT / "results" / "all_results.csv", index=False)
print(f"Загружено {len(results)} результатов из {len(paths)} лабораторных")
display(results.drop(columns=["params"]))''')
    add_pair(nb, "## 2. Эффект улучшенного бейзлайна\n\nСводная таблица содержит основную метрику каждой задачи до и после настройки. Для RMSE меньшее значение лучше, для ROC-AUC — большее.", '''# Строим сводку основной метрики по алгоритму, реализации и этапу настройки.
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
    add_pair(nb, "## 3. Визуальное сравнение\n\nНа графиках показаны улучшенные модели. Для классификации дополнительно приведён F1 по классу `bad`, для регрессии — MAE в МПа, чтобы выбор не опирался на одну метрику.", '''# Строим отдельные графики для несопоставимых задач и сохраняем их в results.
for task in ("classification", "regression"):
    subset = results[(results["task"] == task) & (results["stage"] == "improved")].copy()
    subset["model"] = subset["algorithm"] + "\\n" + subset["implementation"]
    metrics = ["roc_auc", "f1"] if task == "classification" else ["rmse", "mae"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    for axis, metric in zip(axes, metrics):
        colors = ["#3178c6" if value == "sklearn" else "#e76f51" for value in subset["implementation"]]
        axis.bar(subset["model"], subset[metric], color=colors)
        axis.set_title(f"{task}: {metric}, улучшенные модели")
        axis.tick_params(axis="x", rotation=65)
        axis.bar_label(axis.containers[0], fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    figure_path = ROOT / "results" / f"comparison_{task}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Сохранено: {figure_path.relative_to(ROOT)}")''')
    add_pair(nb, "## 4. Итоговые выводы\n\nЛучшие конфигурации определяются по фактической основной метрике. Разница реализаций не является ошибкой: у sklearn и локальных алгоритмов могут различаться критерии остановки, детализация оптимизации и число деревьев (для чистого Python явно ограничено).", '''# Выбираем лидеров по основной метрике и формируем текстовое заключение из рассчитанных значений.
conclusions = []
for task in ("classification", "regression"):
    metric = "roc_auc" if task == "classification" else "rmse"
    ascending = metric == "rmse"
    improved = results[(results["task"] == task) & (results["stage"] == "improved")].sort_values(metric, ascending=ascending)
    best = improved.iloc[0]
    best_sklearn = improved[improved["implementation"] == "sklearn"].iloc[0]
    best_custom = improved[improved["implementation"] == "from_scratch"].iloc[0]
    conclusions.extend([
        f"### {task}",
        f"- Общий лидер по {metric}: **{best['algorithm']} ({best['implementation']})**, значение **{best[metric]:.4f}**.",
        f"- Лучший sklearn: **{best_sklearn['algorithm']}**, {metric} = **{best_sklearn[metric]:.4f}**.",
        f"- Лучшая собственная реализация: **{best_custom['algorithm']}**, {metric} = **{best_custom[metric]:.4f}**.",
    ])
conclusions.extend([
    "### Что учитывать при выборе",
    "Помимо отложенной метрики важны устойчивость CV, стоимость обучения и цена ошибки. "
    "Один train/test-сплит не доказывает статистическую значимость небольших отличий.",
])
display(Markdown("\\n".join(conclusions)))''')
    path = OUT / "06_comparison.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    generated = [build_lab(number, spec) for number, spec in LABS.items()]
    generated.append(build_comparison())
    print("Generated:")
    for path in generated:
        print(path.relative_to(ROOT))
