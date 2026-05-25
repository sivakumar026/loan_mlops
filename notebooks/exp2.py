import mlflow,mlflow.sklearn,dagshub,pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


dagshub.init(repo_owner="sivakumar026",repo_name="loan_mlops",mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sivakumar026/loan_mlops.mlflow")

mlflow.set_experiment("GradientBoosting Hyperparameter Tuning")


with mlflow.start_run(run_name="GradientBoosting_Tuning"):

    gb_model=GradientBoostingClassifier()

    param_grid={
        "n_estimators":[50,100,150],
        "learning_rate":[0.01,0.05,0.1],
        "max_depth":[3,5,7],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1,2]
    }

    df=pd.read_csv("./data/processed/train_processed.csv")

    X=df.drop(columns=["Loan_Status"])
    y=df["Loan_Status"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    grid_search=GridSearchCV(estimator=gb_model,param_grid=param_grid,cv=5,scoring="f1",n_jobs=-1,verbose=2)

    grid_search.fit(X_train,y_train)

    best_model=grid_search.best_estimator_

    print("\nBest Params:\n",grid_search.best_params_)

    y_pred=best_model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)

    print(f"\nAccuracy:{accuracy:.4f} Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}")

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)

    mlflow.sklearn.log_model(sk_model=best_model,name="GradientBoosting_Tuned_Model")

    print("\nModel logged to MLflow successfully")