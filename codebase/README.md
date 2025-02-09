# Hierarchical Intrusion Detection System (HIDS)

A three-stage machine learning pipeline for detecting network intrusions and zero-day attacks using anomaly detection and classification.

## Features

- **Three-Stage Detection**:

  1. **Anomaly Detection**: Identifies suspicious traffic using One-Class SVM/Isolation Forest
  2. **Attack Classification**: Classifies known attacks using Random Forest/XGBoost
  3. **Zero-Day Detection**: Flags novel attacks using anomaly score thresholds

- **Key Components**:
  - Optuna hyperparameter optimization
  - QuantileTransformer preprocessing
  - PCA dimensionality reduction
  - F1-F9 threshold optimization
  - Automated threshold management

## Dataset Setup

### 1. Download Datasets

Download from official source:  
[UNIBO Cybersecurity Datasets (SharePoint)](https://liveunibo-my.sharepoint.com/:f:/g/personal/gokul_shaji_studio_unibo_it/EhMP-n1ACPtHszn92E_Idt0B9iHcor-64VWfXTZEdocWow?e=z2I34U)

**Required Files**:

- `all_benign.parquet` - Normal network traffic
- `all_malicious.parquet` - Attack traffic samples

**Note**: Access may require UNIBO credentials or permission request

### 2. Directory Structure

```bash
project/
├── dataset/
│   ├── all_benign.parquet      # Benign traffic data
│   └── all_malicious.parquet   # Attack traffic data
├── models/                     # Auto-created during training
├── training.py
├── predict.py
├── evaluate.py
└── utils.py
```
