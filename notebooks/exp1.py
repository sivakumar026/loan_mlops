import os,joblib,pandas as pd,dagshub,mlflow,mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


dagshub.init(repo_owner="sivakumar026",repo_name="loan_mlops",mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sivakumar026/loan_mlops.mlflow")

mlflow.set_experiment("Loan Approval Prediction")


def load_data(filepath:str):
    try:return pd.read_csv(filepath)
    except Exception as e:raise Exception(f"Error loading data: {e}")


def evaluate_models(X_train,X_test,y_train,y_test):
    try:
        models={
            "Logistic Regression":LogisticRegression(max_iter=1000),
            "Decision Tree":DecisionTreeClassifier(),
            "Random Forest":RandomForestClassifier(),
            "Gradient Boosting":GradientBoostingClassifier()
        }

        best_model=None
        best_accuracy=0
        best_model_name=""

        for model_name,model in models.items():

            with mlflow.start_run(run_name=model_name):

                model.fit(X_train,y_train)
                y_pred=model.predict(X_test)

                accuracy=accuracy_score(y_test,y_pred)
                precision=precision_score(y_test,y_pred)
                recall=recall_score(y_test,y_pred)
                f1=f1_score(y_test,y_pred)

                print(f"\n{model_name} Accuracy:{accuracy:.4f} Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}")
                print(classification_report(y_test,y_pred))

                mlflow.log_param("model_name",model_name)

                mlflow.log_metric("accuracy",accuracy)
                mlflow.log_metric("precision",precision)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("f1_score",f1)

                mlflow.sklearn.log_model(sk_model=model,name=model_name)

                if accuracy>best_accuracy:
                    best_accuracy=accuracy
                    best_model=model
                    best_model_name=model_name

        print(f"\nBest Model:{best_model_name} Accuracy:{best_accuracy:.4f}")
        return best_model

    except Exception as e:
        raise Exception(f"Error evaluating models: {e}")


def save_model(model,filepath:str):
    try:
        joblib.dump(model,filepath)
        print(f"Model saved at:{filepath}")
    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def main():
    try:
        df=pd.read_csv("./data/processed/train_processed.csv")

        X=df.drop(columns=["Loan_Status"])
        y=df["Loan_Status"]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        print("Data loaded and split completed")

        best_model=evaluate_models(X_train,X_test,y_train,y_test)

        os.makedirs("./models",exist_ok=True)

        save_model(best_model,"./models/best_model.pkl")

        print("Training pipeline completed")

    except Exception as e:
        raise Exception(f"Error in training pipeline: {e}")


if __name__=="__main__":
    main()