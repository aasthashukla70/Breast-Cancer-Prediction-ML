from flask import Flask, render_template, request
import joblib
app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():

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
    prediction = model.predict([features])

    if prediction[0] == 1:
     prediction = "Benign"
    else:
        prediction = "Malignant"

    return render_template("index.html", prediction=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)