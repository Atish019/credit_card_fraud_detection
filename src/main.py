import os
import sys
import mlflow
from sklearn.metrics import roc_auc_score, roc_curve

# Add src folder to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_data, preprocess_data, handle_imbalance, split_data
from src.feature_engineering import feature_engineering
from src.model_training import logistic_regression, decision_tree, random_forest, xgboost
from src.model_evaluation import plot_confusion_matrix, plot_roc_curve

# ✅ Set MLflow to use local directory for logging
mlflow.set_tracking_uri("file:///C:/Users/atish/Desktop/credit card fraud detection/mlruns")

# Load and preprocess data
data = load_data('data/creditcard.csv')
data = feature_engineering(data)
X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Handle imbalanced data with SMOTE
X_train_smote, y_train_smote = handle_imbalance(X_train, y_train)

# Train models and track with MLflow
logistic_regression(X_train_smote, y_train_smote, X_test, y_test)
decision_tree(X_train_smote, y_train_smote, X_test, y_test)
random_forest(X_train_smote, y_train_smote, X_test, y_test)

# Track XGBoost model
xgboost_model = xgboost(X_train_smote, y_train_smote, X_test, y_test)

# Evaluation of XGBoost model
y_pred_xgb = xgboost_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")

y_prob_xgb = xgboost_model.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
plot_roc_curve(fpr_xgb, tpr_xgb, "XGBoost", roc_auc_xgb)
