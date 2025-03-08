from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained severity prediction model
with open("severity_model.pkl", "rb") as file:
    model = pickle.load(file)

# Label encoders for categorical features (must match training data encoding)
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
diet_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3}
exercise_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Daily": 3}
stress_mapping = {"Low": 0, "Medium": 1, "High": 2}
symptom_worsening_mapping = {"No Change": 0, "Slightly Worse": 1, "Much Worse": 2}
symptom_trend_mapping = {"Improving": 0, "Stable": 1, "Worsening": 2}

# API endpoint to predict severity
@app.route("/predict_severity", methods=["POST"])
def predict_severity():
    try:
        data = request.json
        
        # Extract user inputs
        age = int(data.get("age", 30))
        gender = gender_mapping.get(data.get("gender", "Other"), 2)
        diet = diet_mapping.get(data.get("diet", "Sometimes"), 2)
        exercise = exercise_mapping.get(data.get("exercise", "Never"), 0)
        weight_change = int(data.get("weightChange", False))
        smoke_alcohol = int(data.get("smokeAlcohol", False))
        medications = int(data.get("medications", False))
        stress = stress_mapping.get(data.get("stress", "Medium"), 1)
        sleep_issues = int(data.get("sleepIssues", False))
        energy_level = int(data.get("energyLevel", 5))
        symptom_worsening = symptom_worsening_mapping.get(data.get("symptomWorsening", "No Change"), 0)
        symptom_trend = symptom_trend_mapping.get(data.get("symptomTrend", "Stable"), 1)
        consulted_doctor = int(data.get("consultedDoctor", False))
        
        # Convert inputs into a feature vector
        feature_vector = np.array([[
            age, gender, diet, exercise, weight_change, smoke_alcohol,
            medications, stress, sleep_issues, energy_level,
            symptom_worsening, symptom_trend, consulted_doctor
        ]])
        
        # Predict severity (0 = Normal, 1 = Medium, 2 = High)
        severity_prediction = model.predict(feature_vector)[0]
        severity_label = ["Normal", "Medium", "High"][severity_prediction]

        return jsonify({"severity": severity_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run API locally
if __name__ == "__main__":
    app.run(debug=True)
