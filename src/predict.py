import os
import joblib
import numpy as np


def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "model.pkl"
    )

    print("Model Path:", model_path)
    print("Exists:", os.path.exists(model_path))

    model = joblib.load(model_path)
    return model


def load_scaler():
    scaler_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "scaler.pkl"
    )
    scaler = joblib.load(scaler_path)
    return scaler


def predict(features):
    """
    Predict whether the tumor is Benign or Malignant.

    Parameters:
        features (list): List containing 30 feature values.

    Returns:
        str: Prediction result ("Benign" or "Malignant")
    """

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    # Convert input to NumPy array
    features = np.array(features)

    # Reshape to 2D array (1 sample, 30 features)
    features = features.reshape(1, -1)

    # Scale the input
    features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)

    # Convert numerical output to label
    if prediction[0] == 0:
        return "Malignant"
    else:
        return "Benign"