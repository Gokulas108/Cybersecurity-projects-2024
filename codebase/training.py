import optuna
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from utils import (load_network_data, save_artifacts, calculate_anomaly_metrics,
                  determine_classifier_threshold, load_artifacts)
from sklearn.metrics import roc_auc_score

def train_anomaly_detector(X_train, X_val, y_val):
    """Stage 1: Train anomaly detection model with Optuna optimization"""
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_train)
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['ocsvm', 'isoforest'])
        
        if model_type == 'ocsvm':
            params = {
                'nu': trial.suggest_float('nu', 0.01, 0.5),
                'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
            }
            model = OneClassSVM(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'contamination': trial.suggest_float('contamination', 0.01, 0.1)
            }
            model = IsolationForest(**params)
        
        model.fit(X_pca)
        scores = -model.decision_function(pca.transform(X_val))
        return roc_auc_score(y_val, scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # Train final model
    best_params = study.best_params
    if best_params['model_type'] == 'ocsvm':
        model = OneClassSVM(**best_params)
    else:
        model = IsolationForest(**best_params)
    
    model.fit(pca.transform(X_train))
    
    # Save artifacts
    save_artifacts({'pca': pca, 'model': model}, "stage1")
    
    # Calculate thresholds
    val_scores = -model.decision_function(pca.transform(X_val))
    metrics = calculate_anomaly_metrics(y_val, val_scores)
    f4_metrics = metrics[metrics.f_score == 'F4'].iloc[0]
    
    thresholds = {
        'b_threshold': f4_metrics['threshold'],
        'z_threshold': np.quantile(val_scores[y_val == 0], 0.995)
    }
    save_artifacts(thresholds, "thresholds")

def train_attack_classifier(X_train, y_train, X_val, y_val):
    """Stage 2: Train attack classifier with Optuna optimization"""
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb'])
        
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15)
            }
            model = RandomForestClassifier(**params)
        else:
            params = {
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            model = XGBClassifier(**params)
        
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_val)
        return roc_auc_score(y_val, probas, multi_class='ovr', average='weighted')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # Train final model
    best_params = study.best_params
    model = RandomForestClassifier(**best_params) if best_params['model_type'] == 'rf' \
            else XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    save_artifacts(model, "stage2")
    
    # Update confidence threshold
    probas = model.predict_proba(X_val)
    thresholds = load_artifacts("thresholds")
    thresholds['c_threshold'] = determine_classifier_threshold(y_val, probas, model.classes_)
    save_artifacts(thresholds, "thresholds")

if __name__ == "__main__":
    # Load and split data
    (X_benign_train, X_benign_val, X_benign_test,
     X_mal_train, X_mal_test,
     y_benign_train, y_benign_val, y_benign_test,
     y_mal_train, y_mal_test,
     qt) = load_network_data("data/", verbose=True)
    
    # Train Stage 1
    train_anomaly_detector(X_benign_train, X_benign_val, y_benign_val)
    
    # Train Stage 2
    train_attack_classifier(X_mal_train, y_mal_train, X_benign_val, y_benign_val)