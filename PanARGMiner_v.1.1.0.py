import argparse
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Antibiotic Resistance Prediction and Advanced Feature Selection")
    parser.add_argument("-i", "--input", nargs='+', required=True, help="Input CSV file path(s)")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("-t", "--test_size", type=float, default=0.2, help="Test set size (e.g., 0.2 for an 80-20 split)")
    parser.add_argument("-r", "--repeat", type=int, default=5, help="Number of feature selection repetitions")
    parser.add_argument("-m", "--min_occurrence_threshold", type=float, default=0.6, help="Minimum occurrence threshold for feature selection")
    parser.add_argument("-c", "--core", type=int, default=1, help="Number of CPU cores to use (default: 1)")
    
    args = parser.parse_args()
    
    print("\nConfiguration:")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Test set size: {args.test_size}")
    print(f"Number of repetitions: {args.repeat}")
    print(f"Feature occurrence threshold: {args.min_occurrence_threshold}")
    print(f"Number of cores to use: {args.core}")
    
    return args

def normalize_feature_importance(importances):

    scaler = MinMaxScaler(feature_range=(0, 1))
    importances_normalized = scaler.fit_transform(importances.values.reshape(-1, 1))
    return pd.Series(importances_normalized.flatten(), index=importances.index)

def select_features_advanced(X, y, repeat=5, min_occurrence_threshold=0.6, lasso_alpha=0.01, test_size=0.2, core=4):

    feature_score_sum = defaultdict(float)
    feature_run_count = defaultdict(int)
    
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=core),
        'AdaBoost': AdaBoostClassifier(),
        'Lasso': Lasso(alpha=lasso_alpha)
    }
    
    for run in range(repeat):
        print(f"Feature selection run {run+1}/{repeat} ...")
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, stratify=y)
        
        inner_run_features_list = []
        inner_run_importances = []
        for inner in range(2):
            run_features_intersection = None
            run_feature_importance = defaultdict(float)
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                if model_name in ['XGBoost', 'AdaBoost']:
                    importances = pd.Series(model.feature_importances_, index=X_train.columns)
                else:
                    importances = pd.Series(np.abs(model.coef_), index=X_train.columns)
                
                importances_normalized = normalize_feature_importance(importances)
                model_features = set(importances_normalized[importances_normalized > 0].index)
                
                if run_features_intersection is None:
                    run_features_intersection = model_features
                else:
                    run_features_intersection = run_features_intersection.intersection(model_features)
                
                for feature in model_features:
                    run_feature_importance[feature] += importances_normalized.get(feature, 0.0)
            
            inner_run_features_list.append(run_features_intersection)
            inner_run_importances.append(run_feature_importance)
        
        final_run_features = set.intersection(*inner_run_features_list)
        
        X_auc_train, X_auc_test, y_auc_train, y_auc_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
        svc = SVC(kernel='linear', probability=True)
        svc.fit(X_auc_train, y_auc_train)
        y_pred_proba = svc.predict_proba(X_auc_test)[:, 1]
        run_auc = roc_auc_score(y_auc_test, y_pred_proba)
        
        sum_importance = {}
        for feature in final_run_features:
            imp_sum = 0.0
            for inner_run_feature_importance in inner_run_importances:
                if feature in inner_run_feature_importance:
                    imp_sum += inner_run_feature_importance[feature]
            sum_importance[feature] = imp_sum
        
        for feature in final_run_features:
            contribution = sum_importance[feature] * run_auc
            feature_score_sum[feature] += contribution
            feature_run_count[feature] += 1

    denominator = repeat * 2 * 3
    final_feature_scores = {}
    for feature in feature_score_sum:
        final_feature_scores[feature] = feature_score_sum[feature] / denominator
    
    selected_features = {feature: score for feature, score in final_feature_scores.items() 
                         if feature_run_count[feature] >= repeat * min_occurrence_threshold}
    
    selected_features = dict(sorted(selected_features.items(), key=lambda item: item[1], reverse=True))
    return selected_features

def generate_filename(base_name, extension="txt", folder="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def dynamic_label_encoding(series):
    """
    Encode labels into binary (0 and 1).
    Terminates if more than 2 unique labels are found.
    """
    # Standardize strings
    s = series.astype(str).str.strip().str.upper()
    
    # Get unique values excluding null-like strings
    unique_vals = [v for v in s.unique() if v not in ['NAN', 'NONE', '', 'NULL']]
    
    # Validation: Ensure binary classification
    if len(unique_vals) > 2:
        print(f"\nError: Multiple label types detected: {unique_vals}")
        print("This tool supports binary classification only. Please clean your data.")
        sys.exit(1)
    
    if not unique_vals:
        return None, "No valid data found in label column"

    # Encoding Logic
    if s.str.startswith('R').any():
        # Case 1: 'R' exists. R=1, others=0 (including 'I' if present)
        target_mask = s.str.startswith('R')
        strategy = "R-based (R=1, Others=0)"
    elif s.str.startswith('I').any():
        # Case 2: No 'R', but 'I' exists. I=1, others=0
        target_mask = s.str.startswith('I')
        strategy = "I-based (I=1, Others=0)"
    elif len(unique_vals) == 2:
        # Case 3: Neither 'R' nor 'I' found, but 2 categories exist
        target_mask = (s == unique_vals[0])
        strategy = f"Fallback ({unique_vals[0]}=1, {unique_vals[1]}=0)"
    else:
        # Case 4: Only 1 category found
        return None, f"Only one class found: {unique_vals}"

    return target_mask.astype(int), strategy


def main():
    start_time = time.time()
    args = parse_arguments()
    
    os.makedirs(args.output, exist_ok=True)
    summary_file = generate_filename("feature_selection_summary", "txt", folder=args.output)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # --- 進入檔案迴圈 ---
        for input_file in args.input:
            print(f"\nProcessing file: {os.path.basename(input_file)}")
            start_time_per_file = time.perf_counter()
            
            data = pd.read_csv(input_file)
            
            # --- 動態標籤轉換邏輯 ---
            encoded_labels, strategy = dynamic_label_encoding(data['Susceptibility'])
            
            # 如果轉換失敗（例如只有一種類別）則跳過
            if encoded_labels is None:
                msg = f"Skipping {input_file}: {strategy}"
                print(f"  [Warning] {msg}")
                f.write(f"Filename: {os.path.basename(input_file)}\n{msg}\n\n")
                continue # 這裡現在在 for 迴圈內，不會報錯了
            
            data['Susceptibility'] = encoded_labels
            print(f"  [Label Info] Strategy used: {strategy}")
            print(f"  [Label Info] Distribution: \n{data['Susceptibility'].value_counts()}")
            
            # --- 後續處理 ---
            data = shuffle(data).reset_index(drop=True)
            X = data.drop(columns=['Susceptibility', 'Genome ID'], errors='ignore')
            y = data['Susceptibility']
            
            # 分割資料集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, stratify=y
            )
            
            print("  Performing feature selection...")
            # 下面接你原本的特徵選取代碼...
            selected_feature_scores = select_features_advanced(
                X_train, y_train, repeat=args.repeat, min_occurrence_threshold=args.min_occurrence_threshold, 
                lasso_alpha=0.01, test_size=args.test_size, core=args.core
            )
            
            selected_features = list(selected_feature_scores.keys())
            if len(selected_features) == 0:
                print("No valid features selected.")
                f.write("No valid features selected.\n\n")
                continue
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            model = SVC(kernel='linear', probability=True)
            model.fit(X_train_selected, y_train)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            y_pred = model.predict(X_test_selected)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall    = recall_score(y_test, y_pred, zero_division=0)
            f1        = f1_score(y_test, y_pred, zero_division=0)

            end_time_per_file = time.perf_counter()
            elapsed = end_time_per_file - start_time_per_file
            
            f.write(f"Filename: {os.path.basename(input_file)}\n")
            f.write(f"Number of selected features: {len(selected_features)}\n")
            f.write("Selected features:\n")
            for feature in selected_features:
                f.write(f"  - {feature}: {selected_feature_scores[feature]:.4f}\n")
            f.write(f"Model AUC Score: {auc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Training Time (seconds): {elapsed:.2f}\n")
            f.write("\n")
    
    print(f"Feature selection summary saved to: {summary_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()