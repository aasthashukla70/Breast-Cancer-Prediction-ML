import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_data


def train_model(file_path):
    # Get processed data
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Create the model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained model
    joblib.dump(model, "models/model.pkl")
    print("\nModel saved successfully!")

    return model


if __name__ == "__main__":
    train_model("data/raw/breast_cancer.csv")