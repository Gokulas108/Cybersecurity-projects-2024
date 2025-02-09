import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from collections import defaultdict

MODEL_DIR = Path("models")
ZERO_DAY_CLASSES = ['Infiltration', 'Heartbleed']

def load_network_data(data_dir, train_size=100000, val_size=50000, 
                     test_size=30000, file_type="parquet", verbose=True):
    """Load and preprocess network traffic data with advanced sampling"""
    # Load raw data
    benign = pd.read_parquet(f"{data_dir}/all_benign.parquet")
    malicious = pd.read_parquet(f"{data_dir}/all_malicious.parquet")

    # Preprocessing
    for df in [benign, malicious]:
        df.drop(columns=['Timestamp', 'Destination Port'], inplace=True, errors='ignore')

    # Label mapping
    attack_map = {
        'DoS Hulk': '(D)DOS', 'DDoS': '(D)DOS',
        'DoS slowloris': '(D)DOS', 'DoS Slowhttptest': '(D)DOS',
        'DoS GoldenEye': '(D)DOS', 'SSH-Patator': 'Brute Force',
        'FTP-Patator': 'Brute Force', 'Bot': 'Botnet',
        'Web Attack \x96 Brute Force': 'Web Attack',
        'Web Attack \x96 Sql Injection': 'Web Attack',
        'Web Attack \x96 XSS': 'Web Attack',
        'Infiltration': 'Infiltration', 'Heartbleed': 'Heartbleed'
    }
    malicious['Label'] = malicious['Label'].map(attack_map)

    # Split benign data (temporal preservation)
    X_benign = benign.drop(columns=['Label'])
    y_benign = np.zeros(len(benign))
    X_benign_train, X_bt, y_benign_train, y_bt = train_test_split(
        X_benign, y_benign, train_size=train_size, shuffle=False
    )
    X_benign_val, X_benign_test, y_benign_val, y_benign_test = train_test_split(
        X_bt, y_bt, train_size=val_size, test_size=test_size, shuffle=False
    )

    # Process malicious data
    X_malicious = malicious.drop(columns=['Label'])
    y_malicious = malicious['Label']
    train_idx, test_idx = stratified_malicious_split(malicious, y_malicious)
    
    X_malicious_train = X_malicious.iloc[train_idx]
    y_malicious_train = y_malicious.iloc[train_idx]
    X_malicious_test = X_malicious.iloc[test_idx]
    y_malicious_test = y_malicious.iloc[test_idx]

    # Apply QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_benign_train = qt.fit_transform(X_benign_train)
    X_benign_val = qt.transform(X_benign_val)
    X_benign_test = qt.transform(X_benign_test)
    X_malicious_train = qt.transform(X_malicious_train)
    X_malicious_test = qt.transform(X_malicious_test)

    # Print statistics
    if verbose:
        print_data_statistics(
            benign, malicious,
            y_benign_train, y_benign_val, y_benign_test,
            y_malicious_train, y_malicious_test
        )

    return (X_benign_train, X_benign_val, X_benign_test,
            X_malicious_train, X_malicious_test,
            y_benign_train, y_benign_val, y_benign_test,
            y_malicious_train, y_malicious_test,
            qt)

def stratified_malicious_split(df, labels, sample_size=1948, train_ratio=0.7):
    """Custom stratified sampling for attack classes"""
    np.random.seed(42)
    train_idx = []
    test_idx = []
    
    # Isolate zero-day attacks in test set
    zday_mask = labels.isin(ZERO_DAY_CLASSES)
    test_idx.extend(np.where(zday_mask)[0])
    
    # Process other attacks
    for attack in labels[~zday_mask].unique():
        attack_idx = np.where((labels == attack) & ~zday_mask)[0]
        
        if len(attack_idx) < sample_size:
            test_idx.extend(attack_idx)
            continue
            
        np.random.shuffle(attack_idx)
        split = int(len(attack_idx) * train_ratio)
        train_idx.extend(attack_idx[:split])
        test_idx.extend(attack_idx[split:split+sample_size-split])
    
    return np.array(train_idx), np.array(test_idx)

def print_data_statistics(benign_df, malicious_df, y_benign_train, 
                         y_benign_val, y_benign_test, y_mal_train, y_mal_test):
    """Print detailed data distribution statistics"""
    stats = defaultdict(dict)
    
    # Benign stats
    stats[('Benign', 'All')] = {
        'Original': len(benign_df),
        'Train': len(y_benign_train),
        'Validation': len(y_benign_val),
        'Test': len(y_benign_test)
    }
    
    # Malicious stats
    for attack in pd.concat([y_mal_train, y_mal_test]).unique():
        original = (malicious_df['Label'] == attack).sum()
        train = (y_mal_train == attack).sum()
        test = (y_mal_test == attack).sum()
        
        stats[(attack, 'All')] = {
            'Original': original,
            'Train': train,
            'Test': test
        }
    
    print("\nDataset Statistics:")
    print(pd.DataFrame.from_dict(stats, orient='index')
          .sort_index()
          .fillna('-')
          .to_markdown(tablefmt="github", floatfmt=".0f"))

def calculate_anomaly_metrics(y_true, scores):
    """Calculate F1-F9 metrics for anomaly scores"""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    metrics = []
    for beta in range(1, 10):
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
        best_idx = np.argmax(f_beta)
        metrics.append({
            'f_score': f'F{beta}',
            'value': f_beta[best_idx],
            'threshold': thresholds[best_idx],
            'fpr': (scores[y_true == 0] >= thresholds[best_idx]).mean()
        })
    return pd.DataFrame(metrics)

def determine_classifier_threshold(y_true, probas, classes):
    """Find optimal confidence threshold using F1-weighted"""
    thresholds = np.linspace(0.0, 1.0, 100)
    best_f1 = -1
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = []
        for row in probas:
            y_pred.append(classes[np.argmax(row)] if np.max(row) > thresh else 'Unknown')
        
        current_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
            
    return best_threshold

def save_artifacts(obj, name):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"{name}.p", "wb") as f:
        pickle.dump(obj, f)

def load_artifacts(name):
    with open(MODEL_DIR / f"{name}.p", "rb") as f:
        return pickle.load(f)