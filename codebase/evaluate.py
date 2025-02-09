from sklearn.metrics import classification_report
from utils import load_artifacts, load_network_data
from predict import detect_intrusions

def assess_system_performance():
    """Comprehensive system evaluation"""
    # Load data and thresholds
    (_, _, X_test, _, _, _, _, y_mal_test, _, _) = load_network_data("data/", verbose=False)
    thresholds = load_artifacts("thresholds")
    
    # Get predictions
    predictions = detect_intrusions(X_test, **thresholds)
    
    # Prepare labels (combine benign and malicious test sets)
    y_true = np.concatenate([np.zeros(len(X_test) - len(y_mal_test)), y_mal_test])
    y_true = np.where(np.isin(y_true, ['Infiltration', 'Heartbleed']), 'Zero-Day', y_true)
    
    # Clean predictions
    preds_clean = np.where(
        np.isin(predictions, ['Zero-Day']), 'Zero-Day',
        np.where(predictions == 'Benign', 'Benign', 'Known Attack')
    )
    
    # Generate reports
    print("=== Overall Classification Report ===")
    print(classification_report(y_true, preds_clean, digits=4))
    
    print("\n=== Zero-Day Specific Report ===")
    zday_mask = np.isin(y_true, 'Zero-Day')
    print(classification_report(y_true[zday_mask], preds_clean[zday_mask], 
                               target_names=['Zero-Day'], digits=4))

def main():
    print("Starting comprehensive evaluation...")
    assess_system_performance()
    print("Evaluation completed!")

if __name__ == "__main__":
    main()