import os
import joblib
import pandas as pd

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================
# DagsHub + MLflow Setup
# =========================

dagshub.init(
    repo_owner="sivakumar026",
    repo_name="loan_mlops",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/loan_mlops.mlflow"
)

mlflow.set_experiment(
    "Loan Approval Prediction"
)


# =========================
# Load Data
# =========================

def load_data(
    filepath: str
) -> pd.DataFrame:

    try:

        return pd.read_csv(filepath)

    except Exception as e:

        raise Exception(
            f"Error loading data: {e}"
        )


# =========================
# Model Experimentation
# =========================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    try:

        models = {

            "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

            "Decision Tree":
            DecisionTreeClassifier(),

            "Random Forest":
            RandomForestClassifier(),

            "Gradient Boosting":
            GradientBoostingClassifier()

        }

        best_model = None

        best_accuracy = 0

        best_model_name = ""

        print("\nModel Experiment Results\n")

        for model_name, model in models.items():

            # =========================
            # Start MLflow Run
            # =========================

            with mlflow.start_run(
                run_name=model_name
            ):

                # =========================
                # Train Model
                # =========================

                model.fit(
                    X_train,
                    y_train
                )

                # =========================
                # Prediction
                # =========================

                y_pred = model.predict(
                    X_test
                )

                # =========================
                # Metrics
                # =========================

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred
                )

                recall = recall_score(
                    y_test,
                    y_pred
                )

                f1 = f1_score(
                    y_test,
                    y_pred
                )

                # =========================
                # Print Results
                # =========================

                print(
                    f"\n{model_name} Results"
                )

                print(
                    f"Accuracy  : "
                    f"{accuracy:.4f}"
                )

                print(
                    f"Precision : "
                    f"{precision:.4f}"
                )

                print(
                    f"Recall    : "
                    f"{recall:.4f}"
                )

                print(
                    f"F1 Score  : "
                    f"{f1:.4f}"
                )

                print(
                    "\nClassification Report\n"
                )

                print(
                    classification_report(
                        y_test,
                        y_pred
                    )
                )

                print("=" * 50)

                # =========================
                # Log Parameters
                # =========================

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                # =========================
                # Log Metrics
                # =========================

                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "f1_score",
                    f1
                )

                # =========================
                # Log Model
                # =========================

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name
                )

                # =========================
                # Save Best Model
                # =========================

                if accuracy > best_accuracy:

                    best_accuracy = accuracy

                    best_model = model

                    best_model_name = model_name

        # =========================
        # Best Model Details
        # =========================

        print("\nBest Model Details\n")

        print(
            f"Best Model: "
            f"{best_model_name}"
        )

        print(
            f"Best Accuracy: "
            f"{best_accuracy:.4f}"
        )

        return best_model

    except Exception as e:

        raise Exception(
            f"Error evaluating models: {e}"
        )


# =========================
# Save Best Model
# =========================

def save_model(
    model,
    filepath: str
):

    try:

        joblib.dump(
            model,
            filepath
        )

        print(
            f"Model saved at: "
            f"{filepath}"
        )

    except Exception as e:

        raise Exception(
            f"Error saving model: {e}"
        )


# =========================
# Main Function
# =========================

def main():

    try:

        # =========================
        # Paths
        # =========================

        processed_data_path = (
            "./data/processed"
        )

        model_save_path = (
            "./models"
        )

        # Create models folder
        os.makedirs(
            model_save_path,
            exist_ok=True
        )

        # =========================
        # Load Processed Data
        # =========================

        df = load_data(
            os.path.join(
                processed_data_path,
                "train_processed.csv"
            )
        )

        print(
            "Processed data loaded successfully"
        )

        # =========================
        # Features and Target
        # =========================

        X = df.drop(
            columns=["Loan_Status"]
        )

        y = df["Loan_Status"]

        # =========================
        # Train Test Split
        # =========================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print(
            "Train-test split completed"
        )

        # =========================
        # Evaluate Models
        # =========================

        best_model = evaluate_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # =========================
        # Save Best Model
        # =========================

        save_model(
            best_model,
            os.path.join(
                model_save_path,
                "best_model.pkl"
            )
        )

        print(
            "Training pipeline completed"
        )

    except Exception as e:

        raise Exception(
            f"Error in training pipeline: {e}"
        )


if __name__ == "__main__":

    main()