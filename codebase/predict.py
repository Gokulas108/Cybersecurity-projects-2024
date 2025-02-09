import numpy as np
from utils import load_artifacts

def detect_intrusions(X, b_threshold, c_threshold, z_threshold):
    """Three-stage intrusion detection pipeline"""
    # Load models
    stage1 = load_artifacts("stage1")
    stage2 = load_artifacts("stage2")
    
    # Stage 1: Anomaly Detection
    X_pca = stage1['pca'].transform(X)
    scores = -stage1['model'].decision_function(X_pca)
    preds = np.where(scores < b_threshold, "Benign", "Alert").astype(object)
    
    # Stage 2: Attack Classification
    alert_mask = preds == "Alert"
    if np.any(alert_mask):
        probas = stage2.predict_proba(X[alert_mask])
        class_preds = np.where(
            np.max(probas, axis=1) > c_threshold,
            stage2.classes_[np.argmax(probas, axis=1)],
            "Unknown"
        )
        
        # Stage 3: Zero-Day Detection
        unknown_mask = class_preds == "Unknown"
        z_scores = scores[alert_mask][unknown_mask]
        class_preds[unknown_mask] = np.where(
            z_scores < z_threshold, "Benign", "Zero-Day"
        )
        
        preds[alert_mask] = class_preds
    
    return preds

def main():
    """Example prediction usage"""
    from utils import load_network_data
    
    # Load test data
    (_, _, X_test, _, _, _, _, _, _, _) = load_network_data("data/", verbose=False)
    
    # Load thresholds
    thresholds = load_artifacts("thresholds")
    
    # Make predictions
    predictions = detect_intrusions(
        X_test[:1000],  # First 1000 test samples
        thresholds['b_threshold'],
        thresholds['c_threshold'],
        thresholds['z_threshold']
    )
    
    print("Sample predictions:", np.unique(predictions, return_counts=True))

if __name__ == "__main__":
    main()