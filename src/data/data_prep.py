import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:

    try:

        return pd.read_csv(filepath)

    except Exception as e:

        raise Exception(
            f"Error loading data from {filepath}: {e}"
        )


def preprocess_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    try:

        # =========================
        # Drop Loan_ID Column
        # =========================

        if "Loan_ID" in df.columns:

            df = df.drop(
                columns=["Loan_ID"]
            )

        # =========================
        # Handle Missing Values
        # =========================

        for column in df.columns:

            # Numerical Columns
            if df[column].dtype != "object":

                if df[column].isnull().any():

                    mean_value = df[column].mean()

                    df[column] = df[column].fillna(
                        mean_value
                    )

            # Categorical Columns
            else:

                if df[column].isnull().any():

                    mode_value = df[column].mode()[0]

                    df[column] = df[column].fillna(
                        mode_value
                    )

        # =========================
        # Encode Categorical Columns
        # =========================

        label_encoder = LabelEncoder()

        categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns

        for column in categorical_columns:

            df[column] = label_encoder.fit_transform(
                df[column]
            )

        print("Preprocessing completed")

        return df

    except Exception as e:

        raise Exception(
            f"Error during preprocessing: {e}"
        )


def save_data(
    df: pd.DataFrame,
    filepath: str
) -> None:

    try:

        df.to_csv(
            filepath,
            index=False
        )

        print(f"Data saved to {filepath}")

    except Exception as e:

        raise Exception(
            f"Error saving data to {filepath}: {e}"
        )


def main():

    try:

        # =========================
        # Paths
        # =========================

        raw_data_path = "./data/processed"

        processed_data_path = "./data/processed"

        # =========================
        # Create Folder
        # =========================

        os.makedirs(
            processed_data_path,
            exist_ok=True
        )

        # =========================
        # Load Data
        # =========================

        train_data = load_data(
            os.path.join(
                raw_data_path,
                "train.csv"
            )
        )

        test_data = load_data(
            os.path.join(
                raw_data_path,
                "test.csv"
            )
        )

        print("Data loaded successfully")

        # =========================
        # Preprocess Data
        # =========================

        train_processed_data = preprocess_data(
            train_data
        )

        test_processed_data = preprocess_data(
            test_data
        )

        print("Missing values handled")

        # =========================
        # Save Processed Data
        # =========================

        save_data(
            train_processed_data,
            os.path.join(
                processed_data_path,
                "train_processed.csv"
            )
        )

        save_data(
            test_processed_data,
            os.path.join(
                processed_data_path,
                "test_processed.csv"
            )
        )

        print("Data preprocessing completed")

    except Exception as e:

        raise Exception(
            f"An error occurred: {e}"
        )


if __name__ == "__main__":
    main()