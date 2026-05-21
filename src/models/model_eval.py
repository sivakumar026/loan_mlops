import os
import json
import yaml
import joblib
import pandas as pd
import mlflow
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


dagshub.init(repo_owner="sivakumar026",repo_name="loan_mlops",mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sivakumar026/loan_mlops.mlflow")

mlflow.set_experiment("Loan Model Evaluation")


def load_data(filepath:str):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def prepare_data(data:pd.DataFrame):
    try:
        X=data.drop(columns=["Loan_Status"],axis=1)
        y=data["Loan_Status"]
        return X,y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


def load_model(filepath:str):
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


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
        plt.title(f"Confusion Matrix - {model_name}")

        cm_path="reports/confusion_matrix.png"

        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        return {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1
        }

    except Exception as e:
        raise Exception(f"Error in evaluation: {e}")


def save_metrics(metrics:dict,metrics_path:str):
    try:
        with open(metrics_path,"w") as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")


def main():
    try:
        test_data_path="./data/processed/test_processed.csv"
        model_path="models/model.pkl"
        metrics_path="reports/metrics.json"
        model_name="GradientBoosting Loan Model"

        test_data=load_data(test_data_path)
        X_test,y_test=prepare_data(test_data)
        model=load_model(model_path)

        with mlflow.start_run(run_name="Loan_Model_Evaluation"):
            metrics=evaluate_model(model,X_test,y_test,model_name)
            save_metrics(metrics,metrics_path)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            print("Model evaluation completed successfully!")

    except Exception as e:
        raise Exception(f"Error in main: {e}")


if __name__=="__main__":
    main()