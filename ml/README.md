# RouteSense AI: ML module

Python 3.12. Set up:

    py -3.12 -m venv .venv-ml
    .venv-ml\Scripts\python.exe -m pip install -r ml\requirements-ml.txt

Datasets are not in Git. Place them under ml\data\raw\ (CICIDS2017 in cicids2017\, InSDN in insdn\InSDN_DatasetCSV\).

Run order:
1. ml\inspect_labels.py           label inventory
2. ml\data_quality.py             duplicates, inf/NaN checks
3. ml\train_classifiers.py        Stage 1 comparison
4. ml\train_random_forest.py      final Stage 1 model
5. ml\verify_model_reload.py      reload check
6. ml\train_anomaly_detector.py   Stage 2 Isolation Forest
7. ml\verify_stage2_reload.py     reload check
8. ml\explain_predictions.py      SHAP explanations

Tests: .venv-ml\Scripts\python.exe -m pytest ml\tests -v