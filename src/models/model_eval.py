
import json
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


# dagshub.init(
#     repo_owner="sivakumar026",
#     repo_name="loan_mlops",
#     mlflow=True
# )

# mlflow.set_tracking_uri(
#     "https://dagshub.com/sivakumar026/loan_mlops.mlflow"
# )

import os
# Load DagsHub token from environment 
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner="sivakumar026",
repo_name="loan_mlops",
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")



mlflow.set_experiment(
    "Loan_Model_Evaluation"
)


def load_data(filepath:str):

    try:

        return pd.read_csv(filepath)

    except Exception as e:

        raise Exception(f"Error loading data:{e}")


def prepare_data(data:pd.DataFrame):

    try:

        X=data.drop(columns=["Loan_Status"],axis=1)

        y=data["Loan_Status"]

        return X,y

    except Exception as e:

        raise Exception(f"Error preparing data:{e}")


def load_model(filepath:str):

    try:

        return joblib.load(filepath)

    except Exception as e:

        raise Exception(f"Error loading model:{e}")


def evaluate_model(model,X_test,y_test,model_name):

    try:

        with open("params.yaml","r") as file:

            params=yaml.safe_load(file)

        test_size=params["data_collection"]["test_size"]

        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)

        precision=precision_score(y_test,y_pred)

        recall=recall_score(y_test,y_pred)

        f1=f1_score(y_test,y_pred)

        mlflow.log_param("test_size",test_size)

        mlflow.log_metric("accuracy",accuracy)

        mlflow.log_metric("precision",precision)

        mlflow.log_metric("recall",recall)

        mlflow.log_metric("f1_score",f1)

        cm=confusion_matrix(y_test,y_pred)

        os.makedirs("reports",exist_ok=True)

        plt.figure(figsize=(5,5))

        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title(f"ConfusionMatrix-{model_name}")

        cm_path="reports/confusion_matrix.png"

        plt.savefig(cm_path)

        plt.close()

        mlflow.log_artifact(cm_path)

        metrics={
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1
        }

        return metrics

    except Exception as e:

        raise Exception(f"Error in evaluation:{e}")


def save_metrics(metrics:dict,metrics_path:str):

    try:

        with open(metrics_path,"w") as file:

            json.dump(metrics,file,indent=4)

    except Exception as e:

        raise Exception(f"Error saving metrics:{e}")


def save_run_info(run_id:str,model_name:str,filepath:str):

    try:

        run_info={
            "run_id":run_id,
            "model_name":model_name
        }

        with open(filepath,"w") as file:

            json.dump(run_info,file,indent=4)

    except Exception as e:

        raise Exception(f"Error saving run info:{e}")


def main():

    try:

        test_data_path="./data/processed/test_processed.csv"

        model_path="models/model.pkl"

        

        metrics_path="reports/metrics.json"

        run_info_path="reports/run_info.json"

        model_name="loan_prediction_model"

        test_data=load_data(test_data_path)

        X_test,y_test=prepare_data(test_data)

        model=load_model(model_path)

        with mlflow.start_run(run_name="Loan_Model_Evaluation"):

            mlflow.sklearn.autolog()

            metrics=evaluate_model(
                model,
                X_test,
                y_test,
                model_name
            )

            try:

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="loan_model"
                )

                print("loan_model logged successfully")

            except Exception as e:

                print(f"Model logging error:{e}")

            save_metrics(metrics,metrics_path)

            run_id=mlflow.active_run().info.run_id

            save_run_info(
                run_id,
                model_name,
                run_info_path
            )

            mlflow.log_artifact(metrics_path)

            mlflow.log_artifact(run_info_path)

            print("Model evaluation completed successfully!")

    except Exception as e:

        raise Exception(f"Error in main:{e}")


if __name__=="__main__":

    main()