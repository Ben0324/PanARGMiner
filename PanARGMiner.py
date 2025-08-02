import argparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
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
    
    args = parser.parse_args()
    
    print("\nConfiguration:")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Test set size: {args.test_size}")
    print(f"Number of repetitions: {args.repeat}")
    print(f"Feature occurrence threshold: {args.min_occurrence_threshold}")
    
    return args

def normalize_feature_importance(importances):

    scaler = MinMaxScaler(feature_range=(0, 1))
    importances_normalized = scaler.fit_transform(importances.values.reshape(-1, 1))
    return pd.Series(importances_normalized.flatten(), index=importances.index)

def select_features_advanced(X, y, repeat=5, min_occurrence_threshold=0.6, lasso_alpha=0.01, test_size=0.2):

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

def main():
    start_time = time.time()
    args = parse_arguments()
    
    os.makedirs(args.output, exist_ok=True)
    
    summary_file = generate_filename("feature_selection_summary", "txt", folder=args.output)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        for input_file in args.input:
            data = pd.read_csv(input_file)
            data['Susceptibility'] = data['Susceptibility'].map({'Susceptible': 0, 'Resistant': 1})
            data = shuffle(data).reset_index(drop=True)
            
            X = data.drop(columns=['Susceptibility', 'Genome ID'])
            y = data['Susceptibility']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, stratify=y, random_state=42
            )
            
            print("Performing feature selection...")
            selected_feature_scores = select_features_advanced(
                X_train, y_train, repeat=args.repeat, min_occurrence_threshold=args.min_occurrence_threshold, lasso_alpha=0.01, test_size=args.test_size
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
            
            print(f"Processing file: {input_file}")
            f.write(f"Filename: {os.path.basename(input_file)}\n")
            f.write(f"Number of selected features: {len(selected_features)}\n")
            f.write("Selected features:\n")
            for feature in selected_features:
                f.write(f"  - {feature}: {selected_feature_scores[feature]:.4f}\n")
            f.write(f"Model AUC Score: {auc:.4f}\n")
            f.write("\n")
    
    print(f"Feature selection summary saved to: {summary_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
