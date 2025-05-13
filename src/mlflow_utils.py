import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

def start_mlflow_run(model_name, params):
    mlflow.start_run()
    mlflow.set_tag("model_name", model_name)
    mlflow.log_params(params)

def log_model_and_metrics(model, model_name, X_train, X_test, y_train, y_test):
    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log model with signature and input example
    input_example = X_train[:1]
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        input_example=input_example,
        signature=signature
    )

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model {model_name} - Accuracy: {accuracy:.4f}")

def end_mlflow_run():
    mlflow.end_run()
