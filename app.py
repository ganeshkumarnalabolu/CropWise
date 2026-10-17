
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)

# Load the saved model and its expected feature order
saved_artifact = joblib.load(MODEL_PATH)
model = saved_artifact["model"]
FEATURES = saved_artifact["features"]
CLASSES = saved_artifact["classes"]

@app.get("/")
def home():
    return render_template("index.html", prediction=None, error=None)

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


def parse_input(source):
    values = {}

    for feature in FEATURES:
        raw_value = source.get(feature, "")

        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(f"Please enter a value for {feature}.")

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            raise ValueError(f"{feature} must be a valid number.")

        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"{feature} must be a finite number.")

        values[feature] = value

    return pd.DataFrame([values], columns=FEATURES)

def predict_crop(source):
    input_data = parse_input(source)
    prediction = model.predict(input_data)[0]
    return str(prediction)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return jsonify({
            "message": "Submit POST data with N, P, K, temperature, humidity, ph and rainfall."
        }), 200

    if request.is_json:
        source = request.get_json(silent=True)

        if not isinstance(source, dict):
            return jsonify({"error": "A JSON object is required."}), 400

        try:
            crop = predict_crop(source)
            return jsonify({"recommended_crop": crop}), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    try:
        crop = predict_crop(request.form)
        return render_template("index.html", prediction=crop, error=None)
    except ValueError as exc:
        return render_template("index.html", prediction=None, error=str(exc)), 400

@app.errorhandler(500)
def internal_error(error):
    app.logger.exception("Internal application error")
    return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
