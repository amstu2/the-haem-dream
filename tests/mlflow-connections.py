import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tests")

with mlflow.start_run():
    mlflow.log_param("seed", 1024)
    mlflow.log_metric("loss", 0.1)
    mlflow.log_artifact("./test-artifact.txt")