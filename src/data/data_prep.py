import pandas as pd,numpy as np,os
from sklearn.preprocessing import LabelEncoder


def load_data(filepath:str):
    try:return pd.read_csv(filepath)
    except Exception as e:raise Exception(f"Error loading data from {filepath}: {e}")


def preprocess_data(df:pd.DataFrame):
    try:
        if "Loan_ID" in df.columns:df=df.drop(columns=["Loan_ID"])

        for col in df.columns:
            if df[col].dtype!="object":
                if df[col].isnull().any():df[col]=df[col].fillna(df[col].mean())
            else:
                if df[col].isnull().any():df[col]=df[col].fillna(df[col].mode()[0])

        le=LabelEncoder()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col]=le.fit_transform(df[col])

        print("Preprocessing completed")
        return df

    except Exception as e:raise Exception(f"Error during preprocessing: {e}")


def save_data(df:pd.DataFrame,filepath:str):
    try:
        df.to_csv(filepath,index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")


def main():
    try:
        raw_path="./data/raw"
        processed_path="./data/processed"

        os.makedirs(processed_path,exist_ok=True)

        train_df=load_data(f"{raw_path}/train.csv")
        test_df=load_data(f"{raw_path}/test.csv")

        print("Data loaded successfully")

        train_processed=preprocess_data(train_df)
        test_processed=preprocess_data(test_df)

        save_data(train_processed,f"{processed_path}/train_processed.csv")
        save_data(test_processed,f"{processed_path}/test_processed.csv")

        print("Data preprocessing completed")

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__=="__main__":
    main()