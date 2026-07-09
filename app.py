from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature lists for HTML generation
mean_features = [
    ("Mean Radius", "mean_radius"),
    ("Mean Texture", "mean_texture"),
    ("Mean Perimeter", "mean_perimeter"),
    ("Mean Area", "mean_area"),
    ("Mean Smoothness", "mean_smoothness"),
    ("Mean Compactness", "mean_compactness"),
    ("Mean Concavity", "mean_concavity"),
    ("Mean Concave Points", "mean_concave_points"),
    ("Mean Symmetry", "mean_symmetry"),
    ("Mean Fractal Dimension", "mean_fractal_dimension")
]

error_features = [
    ("Radius Error", "radius_error"),
    ("Texture Error", "texture_error"),
    ("Perimeter Error", "perimeter_error"),
    ("Area Error", "area_error"),
    ("Smoothness Error", "smoothness_error"),
    ("Compactness Error", "compactness_error"),
    ("Concavity Error", "concavity_error"),
    ("Concave Points Error", "concave_points_error"),
    ("Symmetry Error", "symmetry_error"),
    ("Fractal Dimension Error", "fractal_dimension_error")
]

worst_features = [
    ("Worst Radius", "worst_radius"),
    ("Worst Texture", "worst_texture"),
    ("Worst Perimeter", "worst_perimeter"),
    ("Worst Area", "worst_area"),
    ("Worst Smoothness", "worst_smoothness"),
    ("Worst Compactness", "worst_compactness"),
    ("Worst Concavity", "worst_concavity"),
    ("Worst Concave Points", "worst_concave_points"),
    ("Worst Symmetry", "worst_symmetry"),
    ("Worst Fractal Dimension", "worst_fractal_dimension")
]


# Home Page
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        mean_features=mean_features,
        error_features=error_features,
        worst_features=worst_features
    )


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    # Read all 30 features in the correct order
    features = [
        float(request.form["mean_radius"]),
        float(request.form["mean_texture"]),
        float(request.form["mean_perimeter"]),
        float(request.form["mean_area"]),
        float(request.form["mean_smoothness"]),
        float(request.form["mean_compactness"]),
        float(request.form["mean_concavity"]),
        float(request.form["mean_concave_points"]),
        float(request.form["mean_symmetry"]),
        float(request.form["mean_fractal_dimension"]),

        float(request.form["radius_error"]),
        float(request.form["texture_error"]),
        float(request.form["perimeter_error"]),
        float(request.form["area_error"]),
        float(request.form["smoothness_error"]),
        float(request.form["compactness_error"]),
        float(request.form["concavity_error"]),
        float(request.form["concave_points_error"]),
        float(request.form["symmetry_error"]),
        float(request.form["fractal_dimension_error"]),

        float(request.form["worst_radius"]),
        float(request.form["worst_texture"]),
        float(request.form["worst_perimeter"]),
        float(request.form["worst_area"]),
        float(request.form["worst_smoothness"]),
        float(request.form["worst_compactness"]),
        float(request.form["worst_concavity"]),
        float(request.form["worst_concave_points"]),
        float(request.form["worst_symmetry"]),
        float(request.form["worst_fractal_dimension"])
    ]

    # Predict
    scaled_features = scaler.transform([features])

    prediction = model.predict(scaled_features)[0]

    prediction = "Benign" if prediction == 1 else "Malignant"

    return render_template(
        "index.html",
        prediction=prediction,
        mean_features=mean_features,
        error_features=error_features,
        worst_features=worst_features
    )


if __name__ == "__main__":
    app.run(debug=True)