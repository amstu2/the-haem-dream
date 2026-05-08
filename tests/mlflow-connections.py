import mlflow

MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"

experiment_name = "tests"
with mlflow.start_run():
    mlflow.log_param("seed", 1024)
    mlflow.log_metric("loss", 0.1)
    mlflow.log_artifact("./test-artifact.txt")