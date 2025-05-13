import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Directory to save plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot Confusion Matrix and log to MLflow
def plot_confusion_matrix(y_test, y_pred, model_name, save_path=PLOT_DIR, log_to_mlflow=True):
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Non-Fraudulent', 'Fraudulent']

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)

    matrix_labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            label = matrix_labels[i][j]
            ax.text(j + 0.5, i + 0.5, f'{label} = {value}', 
                    ha='center', va='center', color='black', fontsize=12)

    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    filepath = os.path.join(save_path, f'{model_name}_confusion_matrix.png')
    plt.savefig(filepath)
    plt.close()

    if log_to_mlflow:
        mlflow.log_artifact(filepath, artifact_path="plots")

# Plot ROC Curve and log to MLflow
def plot_roc_curve(fpr, tpr, model_name, roc_auc, save_path=PLOT_DIR, log_to_mlflow=True):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')

    filepath = os.path.join(save_path, f'{model_name}_roc_curve.png')
    plt.savefig(filepath)
    plt.close()

    if log_to_mlflow:
        mlflow.log_artifact(filepath, artifact_path="plots")
