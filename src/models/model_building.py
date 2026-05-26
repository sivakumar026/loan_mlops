import os
import yaml
import joblib
import pandas as pd

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# Load Parameters
# =========================
def load_params(params_path:str):
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")


# Load Data
# =========================
def load_data(data_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")


# =========================
# Prepare Data
# =========================
def prepare_data(data:pd.DataFrame):
    try:
        X=data.drop(columns=["Loan_Status"],axis=1)
        y=data["Loan_Status"]
        return X,y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


# =========================
# Train Model
# =========================
def train_model(X_train,y_train,params):
    try:
        model=GradientBoostingClassifier(
            n_estimators=params["model_building"]["n_estimators"],
            learning_rate=params["model_building"]["learning_rate"],
            max_depth=params["model_building"]["max_depth"],
            min_samples_split=params["model_building"]["min_samples_split"],
            min_samples_leaf=params["model_building"]["min_samples_leaf"],
            random_state=params["model_building"]["random_state"]
        )

        model.fit(X_train,y_train)
        return model

    except Exception as e:
        raise Exception(f"Error training model: {e}")


# =========================
# Save Model
# =========================
def save_model(model,model_name:str)->None:
    try:
        os.makedirs("models",exist_ok=True)
        joblib.dump(model,model_name)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")


# =========================
# Main Function
# =========================
def main():
    try:
        params_path="params.yaml"
        data_path="./data/processed/train_processed.csv"
        model_name="models/model.pkl"

        dagshub.init(
            repo_owner="sivakumar026",
            repo_name="loan_mlops",
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/sivakumar026/loan_mlops.mlflow"
        )

        params=load_params(params_path)

        mlflow.set_experiment(
            "GradientBoosting Production Model"
        )

        train_data=load_data(data_path)
        print("Dataset loaded successfully")

        X,y=prepare_data(train_data)

        X_train,X_test,y_train,y_test=train_test_split(
            X,
            y,
            test_size=params["data_collection"]["test_size"],
            random_state=params["model_building"]["random_state"]
        )

        print("Train-test split completed")

        with mlflow.start_run(run_name="GradientBoosting_Training"):

            model=train_model(X_train,y_train,params)

            y_pred=model.predict(X_test)

            accuracy=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            mlflow.log_param("n_estimators",params["model_building"]["n_estimators"])
            mlflow.log_param("learning_rate",params["model_building"]["learning_rate"])
            mlflow.log_param("max_depth",params["model_building"]["max_depth"])
            mlflow.log_param("min_samples_split",params["model_building"]["min_samples_split"])
            mlflow.log_param("min_samples_leaf",params["model_building"]["min_samples_leaf"])
            mlflow.log_param("random_state",params["model_building"]["random_state"])

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1_score",f1)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="GradientBoosting_Model"
            )

            save_model(model,model_name)

            print("Model trained and saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    main()