# PanARGMiner: Pan-genomic Antimicrobial Resistance Gene Miner

Feature Selection for AMR using XGBoost, AdaBoost, Lasso and SVM*
This Python script implements a robust framework for performing antibiotic resistance feature selection across pan-genomic datasets. It combines multiple machine learning models (XGBoost, AdaBoost, Lasso) and evaluates the selected features with an SVM classifier using AUC score as performance measure.

---

##  Features

- Efficient at extracting key predictive features from large-scale pan-genomic datasets
- Multi-model feature selection: **XGBoost**, **AdaBoost**, **Lasso**
- Repeated intersection-based feature voting
- AUC-weighted scoring of features
- SVM-based evaluation of selected feature sets
- Outputs are automatically timestamped for reproducibility

---

##  Input Format

Each input CSV should contain:

| Column Name      | Description                                    |
|------------------|------------------------------------------------|
| `Genome ID`      | Unique identifier of the sample                |
| `ClusterXXXX`    | Binary feature columns (0 or 1)                |
| `Susceptibility` | Phenotype label (`Susceptible` or `Resistant`) |

Example:

| Genome ID | Cluster0 | Cluster1 | Cluster2 | Cluster3 | Cluster4 | Susceptibility |
|-----------|----------|----------|----------|----------|----------|----------------|
| 123456.1 |    0     |    0     |    1     |    0     |    1     |   Resistant     |
| 123456.2 |    1     |    0     |    0     |    0     |    1     |   Susceptible |

---

##  Command-Line Arguments

| Flag / Option                      | Description                                                     |
|-----------------------------------|-----------------------------------------------------------------|
| `-i`, `--input`                    | One or more input CSV file paths                                |
| `-o`, `--output`                   | Directory to save results                                       |
| `-t`, `--test_size`                | Proportion of test set (default: `0.2`)                         |
| `-r`, `--repeat`                   | Number of repetitions for feature selection (default: `5`)      |
| `-m`, `--min_occurrence_threshold` | Minimum frequency threshold to retain features (default: `0.6`) |

---

##  Example Usage

```bash
python PanARGMiner.py \
  -i ecoli_data.csv \
  -o ./results \
  -t 0.2 \
  -r 5 \
  -m 0.6
```

---

##  Methodology

1. **Preprocessing**
   - Converts `Susceptibility` to binary (Resistant=1, Susceptible=0)
   - Splits dataset using stratified `train_test_split`

2. **Model Training**
   - Runs XGBoost, AdaBoost, and Lasso
   - Normalizes importance via Min-Max scaling
   - Intersects top features from all models

3. **Feature Voting**
   - Repeats process multiple times
   - Computes occurrence frequency for each feature
   - Scores features using AUC × importance

4. **Final Evaluation**
   - Selects features that appear ≥ threshold ratio of runs
   - Trains final SVM (linear kernel) on selected features
   - Reports AUC on test set

---

##  Output

Output summary is saved in the output folder, named with a timestamp, e.g.:

```
feature_selection_summary_20250511_153000.txt
```

Contents include:

- Filename
- Number of selected features
- Feature scores
- Model AUC score

Example:

```text
Filename: ecoli_data.csv
Number of selected features: 4
Selected features:
  - Cluster2: 0.6413
  - Cluster8: 0.3494
  - Cluster26: 0.2749
  - Cluster51: 0.0462
  ...
Model AUC Score: 0.9338
Precision: 1.0000
Recall: 0.9172
F1-score: 0.9568
Training Time (seconds): 161.63
```

---

##  Dependencies

Install required packages for Python 3.12:

```bash
pip install pandas==2.3.1 numpy==2.3.1 xgboost==3.0.2 scikit-learn==1.7.1 python-dateutil==2.9.0.post0 pytz==2025.2
```

---

##  License
This project is licensed under the MIT License.

---

## Citation

If you use this script in your research, please cite or acknowledge as follows:

PanARGMiner:  An Advanced Cross-Validated Feature Selection Framework for Extracting Key Antimicrobial Resistance Proteins from Large-Scale Pan-genomic Datasets (manuscript currently under submission)
