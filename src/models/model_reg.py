import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub


dagshub.init(
    repo_owner="sivakumar026",
    repo_name="loan_mlops",
    mlflow=True
)

mlflow.set_experiment(
    "Final_Model"
)

mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/loan_mlops.mlflow"
)


reports_path="reports/run_info.json"

with open(reports_path,"r") as file:

    run_info=json.load(file)


run_id=run_info["run_id"]

model_name=run_info["model_name"]


client=MlflowClient()


model_uri=f"runs:/{run_id}/loan_model"


registered_model=mlflow.register_model(
    model_uri,
    model_name
)


model_version=registered_model.version


new_stage="Staging"


client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)


print(
    f"Model {model_name} "
    f"version {model_version} "
    f"transitioned to {new_stage} stage."
)