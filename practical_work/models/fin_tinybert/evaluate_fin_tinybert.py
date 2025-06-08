from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr, spearmanr
import numpy as np

def bin_and_classify(scores, pos_thresh=0.3, neg_thresh=-0.3):
    return [1 if s > pos_thresh else -1 if s < neg_thresh else 0 for s in scores]

def evaluate_regression_and_classification(y_true, y_pred, model_name="Model"):
    y_pred_class = bin_and_classify(y_pred)
    y_true_class = [int(round(s)) for s in y_true]

    y_true_bin = label_binarize(y_true_class, classes=[-1, 0, 1])
    y_pred_bin = label_binarize(y_pred_class, classes=[-1, 0, 1])

    results = {
        "Model": model_name,
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Pearson_r": pearsonr(y_true, y_pred)[0],
        "Spearman_rho": spearmanr(y_true, y_pred)[0],
        "Accuracy": accuracy_score(y_true_class, y_pred_class),
        "Precision": precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0),
        "Recall": recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0),
        "F1": f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0),
        "ROC_AUC": roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovo'),
        "Cohen_Kappa": cohen_kappa_score(y_true_class, y_pred_class),
        "Confusion_Matrix": confusion_matrix(y_true_class, y_pred_class)
    }

    print(f"\n=== Evaluation: {model_name} ===")
    for k, v in results.items():
        if k == "Confusion_Matrix":
            print(f"{k}:\n{v}")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
    return results

results = evaluate_regression_and_classification(y_true, y_pred, model_name="Fine-Tuned TinyBERT")  # or "Base TinyBERT"
