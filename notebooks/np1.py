import dagshub
import mlflow

dagshub.init(
    repo_owner='sivakumar026',
    repo_name='loan_mlops',
    mlflow=True
)
mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/loan_mlops.mlflow"
)
with mlflow.start_run():

    mlflow.log_param("model", "RandomForest")

    mlflow.log_metric("accuracy", 0.92)

    mlflow.log_metric("f1_score", 0.89)

   