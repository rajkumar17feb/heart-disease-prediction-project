from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data in the right order
        features = [
            int(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            int(request.form["trestbps"]),
            int(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            int(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]

        # Convert to numpy array for prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        # Convert prediction to readable text
        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
