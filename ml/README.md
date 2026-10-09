Run order:
1. ml\reports\inspect_labels.py        label inventory
2. ml\reports\data_quality.py          duplicates, inf/NaN checks
3. ml\models\train_classifiers.py      Stage 1 comparison
4. ml\models\train_random_forest.py    final Stage 1 model
5. ml\models\verify_model_reload.py    reload check
6. ml\models\train_anomaly_detector.py Stage 2 Isolation Forest
7. ml\models\verify_stage2_reload.py   reload check
8. ml\models\explain_predictions.py    SHAP explanations
9. ml\reports\simple_summary.py        plain-language summary
10. ml\reports\make_report_image.py    report image
