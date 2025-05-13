import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import os
import pickle
# Track model with MLflow (with signature & input example)
def track_model_with_mlflow(model, model_name, params, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Add signature and input_example
        input_example = X_train[:1]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            input_example=input_example,
            signature=signature
        )

        print(f"{model_name} - Accuracy: {accuracy:.4f}")
        return model

# Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    params = {"max_iter": 1000, "random_state": 42}
    return track_model_with_mlflow(model, "Logistic Regression", params, X_train, y_train, X_test, y_test)

# Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    params = {"max_depth": 5, "random_state": 42}
    return track_model_with_mlflow(model, "Decision Tree", params, X_train, y_train, X_test, y_test)

# Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    params = {"n_estimators": 100, "random_state": 42}
    return track_model_with_mlflow(model, "Random Forest", params, X_train, y_train, X_test, y_test)

# XGBoost
def xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}

    trained_model = track_model_with_mlflow(model, "XGBoost", params, X_train, y_train, X_test, y_test)

    # Save the trained XGBoost model as a Pickle file
    os.makedirs("artifacts/models", exist_ok=True)
    model_save_path = "artifacts/models/xgboost_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(trained_model, f)

    print(f"XGBoost model saved as '{model_save_path}'")
    return trained_model