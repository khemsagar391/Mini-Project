\# RouteSense AI: Stage 1 Supervised Classifier Comparison



\*\*Dataset:\*\* CICIDS2017 (TrafficLabelling version, 8 CSV files, 2,830,743 flows after removing 288,602 empty padding rows)

\*\*Script:\*\* `ml/train\_classifiers.py`

\*\*Features (schema v1.0):\*\* total packets, total bytes, duration, packets/s, bytes/s, average packet size

\*\*Hardware:\*\* XGBoost on NVIDIA RTX 5050 (CUDA); all other models on CPU



\## 1. Method



\- \*\*Split:\*\* within each label, flows are sorted by time and cut 70% train, 15% validation, 15% test. Every label appears in all three splits.

\- \*\*Model selection:\*\* uses validation data only.

\- \*\*Test data:\*\* reported for reference. It was not used to choose a model.

\- \*\*Evaluation sample:\*\* validation and test scoring use up to 20,000 flows per label (85,147 validation and 85,152 test flows).

\- \*\*Training sample:\*\* Logistic Regression, Decision Tree, Random Forest and XGBoost train on all 1,981,513 training flows. KNN and SVM train on a capped sample of 16,432 flows (up to 1,500 per label), because they do not scale to the full set on this laptop.



\## 2. Validation results (used for selection)



| Model | Train flows | Fit (s) | Validation predict (s) | Macro-F1 | PR-AUC (macro) | Accuracy | BENIGN false-positive rate |

|---|---|---|---|---|---|---|---|

| XGBoost (GPU) | 1,981,513 | 48.7 | 1.15 | \*\*0.678\*\* | 0.779 | 0.953 | 0.129 |

| Random Forest | 1,981,513 | 25.3 | 0.47 | 0.671 | \*\*0.777\*\* | 0.959 | 0.104 |

| Decision Tree | 1,981,513 | 9.9 | 0.02 | 0.665 | 0.670 | 0.952 | 0.117 |

| KNN | 16,432 | 0.0 | 0.50 | 0.612 | 0.564 | 0.862 | 0.207 |

| SVM | 16,432 | 3.1 | 89.61 | 0.434 | 0.378 | 0.632 | 0.774 |

| Logistic Regression | 1,981,513 | 237.4 | 0.03 | 0.283 | 0.389 | 0.590 | 0.878 |



\*\*Highest validation macro-F1:\*\* XGBoost, narrowly ahead of Random Forest. The two are effectively tied on macro-F1 and PR-AUC.



\## 3. Test results (reference only)



| Model | Macro-F1 | PR-AUC (macro) | Accuracy | BENIGN false-positive rate |

|---|---|---|---|---|

| Random Forest | 0.605 | 0.645 | 0.882 | 0.161 |

| XGBoost | 0.587 | 0.618 | 0.895 | 0.181 |

| Decision Tree | 0.591 | 0.560 | 0.877 | 0.174 |

| KNN | 0.492 | 0.455 | 0.767 | 0.259 |

| SVM | 0.408 | 0.349 | 0.562 | 0.765 |

| Logistic Regression | 0.248 | 0.355 | 0.510 | 0.873 |



On test data, Random Forest scores slightly higher than XGBoost on macro-F1 and PR-AUC. This is the expected kind of gap for two closely matched models, and it is one reason the final choice should be discussed, not decided by one number.



\## 4. Per-class results (XGBoost, test sample)



| Class | Precision | Recall | F1 | Test flows | Note |

|---|---|---|---|---|---|

| BENIGN | 0.833 | 0.819 | 0.826 | 20,000 | |

| DDoS | 0.940 | 0.996 | 0.967 | 19,205 | |

| DoS GoldenEye | 0.921 | 0.982 | 0.951 | 1,544 | |

| DoS Hulk | 0.975 | 0.814 | 0.887 | 20,000 | |

| DoS Slowhttptest | 0.946 | 0.813 | 0.875 | 825 | |

| DoS slowloris | 0.735 | 0.399 | 0.517 | 870 | |

| FTP-Patator | 0.971 | 0.993 | 0.982 | 1,191 | |

| PortScan | 0.941 | 0.991 | 0.965 | 20,000 | |

| SSH-Patator | 0.525 | 0.827 | 0.642 | 885 | |

| Bot | 0.000 | 0.000 | 0.000 | 295 | Not detected |

| Web Attack - Brute Force | 0.280 | 0.322 | 0.299 | 227 | Weak |

| Web Attack - XSS | 0.122 | 0.510 | 0.197 | 98 | Weak |

| Heartbleed | 1.000 | 0.500 | 0.667 | 2 | Too few rows to judge |

| Infiltration | 0.000 | 0.000 | 0.000 | 6 | Too few rows to judge |

| Web Attack - Sql Injection | 0.019 | 0.500 | 0.037 | 4 | Too few rows to judge |



\## 5. Limitations



1\. \*\*Not a true session split.\*\* Flows from one attack run can fall on both sides of a time cut, so test scores may be somewhat optimistic.

2\. \*\*SYN, UDP and ICMP floods are not labelled\*\* in either dataset. They fall inside the DoS and DDoS classes, so the project's three flood targets are not evaluated separately.

3\. \*\*Low-support classes.\*\* Heartbleed (2 test flows), Infiltration (6) and Web Attack - Sql Injection (4) are too small for reliable scores. They are candidates for held-out testing in Stage 2.

4\. \*\*Weak classes.\*\* Bot is not detected at all on test data. The web attacks are weak, and DoS slowloris is partly missed.

5\. \*\*Unequal training samples.\*\* KNN and SVM used 16,432 training flows, so their scores are not directly comparable with the four full-data models.

6\. \*\*Unequal hardware.\*\* XGBoost ran on the GPU and the others on the CPU, so timings are not a fair speed race.

7\. \*\*Average packet size\*\* is computed as total bytes divided by total packets. The raw CICIDS2017 column differs by about 5 to 10%, so it is not used.

8\. \*\*False-positive rate.\*\* The best BENIGN false-positive rate on validation is about 10%. A live controller acting on single predictions would block healthy hosts, so the model's output must go through the controller's confirmation rules before any mitigation.



\## 6. Status



| Item | Status |

|---|---|

| Feature contract v1.0 and loader (19 unit tests) | TESTED |

| Stage 1 comparison of six classifiers | RUN, final model not yet chosen |

| Stage 2 Isolation Forest | NOT STARTED |

| SHAP explanations | NOT STARTED |

| Model saving and reload check | NOT STARTED |

| InSDN cross-dataset test | NOT STARTED |



\## 7. Files



\- `ml/metrics/stage1\_comparison.csv`: full comparison table

\- `ml/metrics/stage1\_per\_class\_test.csv`: per-class results for XGBoost

\- `ml/train\_classifiers.py`: the script that produced these results

