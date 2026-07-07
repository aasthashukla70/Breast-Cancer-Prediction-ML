import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path):
    """
    Load dataset, split into train/test sets,
    apply feature scaling, save scaler,
    and return processed data.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Validate target column
    if "target" not in df.columns:
        raise ValueError("Target column 'target' not found in dataset.")

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    # Learn scaling parameters from training data
    X_train = scaler.fit_transform(X_train)

    # Apply the same scaling to test data
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/raw/breast_cancer.csv"
    )

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)