import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# =========================
# MLflow Experiment
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
    "GradientBoosting Hyperparameter Tuning"
)

# =========================
# Start MLflow Run
# =========================

with mlflow.start_run(
    run_name="GradientBoosting_Tuning"
):

    # =========================
    # Model
    # =========================

    gb_model = GradientBoostingClassifier()

    # =========================
    # Hyperparameter Grid
    # =========================

    param_grid = {

        "n_estimators":[50,100,150],

        "learning_rate":[0.01,0.05,0.1],

        "max_depth":[3,5,7],

        "min_samples_split":[2,5],

        "min_samples_leaf":[1,2]
    }

    # =========================
    # Grid Search
    # =========================

    grid_search = GridSearchCV(

        estimator=gb_model,

        param_grid=param_grid,

        cv=5,

        scoring="f1",

        n_jobs=-1,

        verbose=2
    )

    # =========================
    # Train
    # =========================

    
    df = pd.read_csv(
        "./data/processed/train_processed.csv"
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

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    grid_search.fit(
        X_train,
        y_train
    )

    # =========================
    # Best Model
    # =========================

    best_gb_model = grid_search.best_estimator_

    print("\nBest Parameters:\n")

    print(
        grid_search.best_params_
    )

    # =========================
    # Prediction
    # =========================

    y_pred = best_gb_model.predict(
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
        "\nTuned Gradient Boosting Results\n"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    # =========================
    # Log Parameters
    # =========================

    mlflow.log_params(
        grid_search.best_params_
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
        sk_model=best_gb_model,
        name="GradientBoosting_Tuned_Model"
    )

    print(
        "\nModel logged to MLflow successfully"
    )