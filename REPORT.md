# Лабораторная работа 4
## Данные и подготовка
- Split: `german.csv` (1000 строк, 20 признаков, таргет `Creditability`, доля класса `1` — 70%). Стратифицированное разбиение 80/20 (`random_state=42`). Препроцессинг обёрнут в `Pipeline` и обучается только на `train`.
- Новые признаки: отдельно для `train` и `test` добавлены 3 признака:
  - `monthly_payment_burden` = `Credit_Amount / Duration_of_Credit_monthly`
  - `log_credit_amount` = `log1p(Credit_Amount)`
  - `credit_to_age_ratio` = `Credit_Amount / Age_years`
- Преобразования: 13 категориальных признаков (в файле записаны числами) выделены отдельно — заполнение модой + `OneHotEncoder(handle_unknown="ignore")`. Числовые — медиана + `RobustScaler` (для устойчивости к выбросам и корректной работы `MLPClassifier`).

## Гипотезы и эксперименты
Все модели проверялись по 5-фолдовой стратифицированной кросс-валидации (`StratifiedKFold`, `random_state=42`) на `train` (800 объектов):
1. `RandomForest_baseline` (`n_estimators=200`) — базовая модель без ограничения глубины (проверка склонности к переобучению).
2. `RandomForest_tuned` (`n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced_subsample"`) — ограничение глубины и балансировка весов для борьбы с переобучением и дисбалансом 70/30.
3. `ExtraTrees_tuned` (`n_estimators=300`, `max_depth=10`, `min_samples_leaf=4`) — проверка сверхслучайных деревьев для снижения дисперсии.
4. `HistGradientBoosting` (`max_iter=200`, `learning_rate=0.05`, `max_depth=4`, `min_samples_leaf=20`, `l2_regularization=1.0`) — бустинг с неглубокими деревьями и L2-регуляризацией.
5. `MLP_64_32` (`hidden_layer_sizes=(64, 32)`, `relu`, `alpha=1.0`, `early_stopping=True`) — проверка двухслойной нейросети с регуляризацией и ранней остановкой.
6. `SoftVoting(HistGradientBoosting + RandomForest + MLP)` (`voting="soft"`, веса `(2, 2, 1)`) — усреднение вероятностей трёх разных семейств моделей.

## Метрики
Критерий выбора модели — CV ROC-AUC (качество ранжирования вероятностей независимо от порога).

| Модель | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest_baseline | 0.7757 | 0.8349 | 0.8200 | 0.8171 | 0.9571 | 0.8816 |
| RandomForest_tuned | 0.7882 | 0.8471 | 0.7800 | 0.8478 | 0.8357 | 0.8417 |
| ExtraTrees_tuned | 0.7734 | 0.8250 | 0.7750 | 0.7778 | 0.9500 | 0.8553 |
| HistGradientBoosting | 0.7753 | 0.8152 | 0.7900 | 0.8224 | 0.8929 | 0.8562 |
| MLP_64_32 | 0.6503 | 0.8173 | 0.7950 | 0.7861 | 0.9714 | 0.8690 |
| SoftVoting(HistGradientBoosting + RandomForest + MLP) | 0.7873 | 0.8358 | 0.7850 | 0.8170 | 0.8929 | 0.8532 |

## Выбранная модель
Выбрана `RandomForest_tuned` (`n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced_subsample"`):
- Лидер по критерию отбора (CV ROC-AUC = 0.7882) и на тесте (Test ROC-AUC = 0.8471).
- В отличие от базового леса, который завышал Recall (0.9571) ценой низкого Precision (0.8171), настроенный лес дал сбалансированные Precision (0.8478) и Recall (0.8357).
- `SoftVoting` стал вторым (CV ROC-AUC = 0.7873), уступив из-за просадки `MLP` на фолдах.

## Выводы и ограничения
- Улучшило результат: разделение категориальных и числовых признаков, признак ежемесячной нагрузки, ограничение глубины деревьев (`max_depth=8`) и `class_weight="balanced_subsample"` (+0.0125 к CV ROC-AUC).
- Ухудшило результат: `MLP_64_32` с `early_stopping=True` (CV ROC-AUC = 0.6503) — при 800 объектах в `train` отщипывание 15% под валидацию внутри фолдов оставляло слишком мало данных для обучения.
- Ограничения: малый объём выборки (1000 строк) даёт разброс оценок на тесте (200 строк); порог бинаризации зафиксирован на 0.5 без бизнес-оптимизации.

## Воспроизводимость
- Seed: везде `random_state=42` (`train_test_split`, `StratifiedKFold`, модели); в `run_ensemble_experiment` используется `data.random_seed`.
- Запуск: контрактная функция `run_ensemble_experiment(data)` реализована в `ensemble_models.py`; пересчёт всех экспериментов в `experiment_results.json` — `python run_experiments.py`.
