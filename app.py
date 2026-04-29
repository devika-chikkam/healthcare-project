from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file
from flask import make_response
import io

from database import create_table, connect_db

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# -------------------------------
# Initialize Database
# -------------------------------
create_table()

# -------------------------------
# Password Validation
# -------------------------------
def is_valid_password(password):
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*[@$!%*?&]).{6,}$'
    return re.match(pattern, password)

# -------------------------------
# Load ML Model
# -------------------------------
model = joblib.load('models/model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    return "Healthcare AI Backend Running!"

# -------------------------------
# REGISTER API
# -------------------------------
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json

        name = data['name']
        email = data['email']
        password = data['password']

        # Password validation
        if not is_valid_password(password):
            return jsonify({
                "error": "Password must contain uppercase, lowercase, special character"
            }), 400

        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )

        conn.commit()
        conn.close()

        return jsonify({"message": "User registered successfully"})

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# LOGIN API
# -------------------------------
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json

        email = data['email']
        password = data['password']

        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({"message": "Login successful"})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# PREDICTION API
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        age = data['age']
        gender = data['gender']
        symptoms = data['symptoms']

        # Encode gender
        gender = 0 if gender == "Male" else 1

        # Create input vector
        input_data = [0] * len(feature_names)
        input_dict = dict(zip(feature_names, input_data))

        input_dict['age'] = age
        input_dict['gender'] = gender

        # Set symptoms
        for symptom in symptoms:
            if symptom in input_dict:
                input_dict[symptom] = 1

        final_input = np.array([list(input_dict.values())])

        # Scale
        final_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        # Confidence
        probabilities = model.predict_proba(final_input)
        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": predicted_disease,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
# -------------------------------
# Download Report API
# -------------------------------
@app.route('/download-report', methods=['POST', 'OPTIONS'])
def download_report():

    # ✅ HANDLE PREFLIGHT REQUEST FIRST
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    try:
        data = request.json

        disease = data['prediction']
        confidence = data['confidence']
        age = data['age']
        gender = data['gender']
        symptoms = data['symptoms']

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("Health Prediction Report", styles['Title']))
        content.append(Spacer(1, 12))

        content.append(Paragraph(f"Age: {age}", styles['Normal']))
        content.append(Paragraph(f"Gender: {gender}", styles['Normal']))
        content.append(Paragraph(f"Symptoms: {', '.join(symptoms)}", styles['Normal']))
        content.append(Spacer(1, 12))

        content.append(Paragraph(f"Predicted Disease: {disease}", styles['Heading2']))
        content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

        doc.build(content)

        buffer.seek(0)

        response = send_file(
            buffer,
            as_attachment=True,
            download_name="report.pdf",
            mimetype='application/pdf'
        )

        # ✅ ADD HEADERS HERE TOO
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Gen AI Explanation API
# -------------------------------
@app.route('/explain', methods=['POST'])
def explain():
    data = request.json
    disease = data['disease']

    # -------------------------------
    # YOUR DATA-BASED KNOWLEDGE
    # -------------------------------
    explanations = {
        "Dengue": {
            "about": "Dengue is a mosquito-borne viral infection.",
            "causes": "Spread by Aedes mosquitoes.",
            "symptoms": "Fever, joint pain, headache, nausea.",
            "precautions": "Avoid mosquito bites, use repellents."
        },
        "COVID-19": {
            "about": "COVID-19 is a viral respiratory disease.",
            "causes": "Caused by SARS-CoV-2 virus.",
            "symptoms": "Fever, cough, breathlessness.",
            "precautions": "Wear masks, maintain hygiene."
        },
        "Flu": {
            "about": "Flu is a viral infection affecting the respiratory system.",
            "causes": "Influenza virus.",
            "symptoms": "Fever, cough, fatigue.",
            "precautions": "Rest, hydration, vaccination."
        },
        "Malaria": {
            "about": "Malaria is caused by parasites transmitted by mosquitoes.",
            "causes": "Plasmodium parasite.",
            "symptoms": "Fever, chills, sweating.",
            "precautions": "Use mosquito nets, medication."
        },
        "Common Cold": {
            "about": "Common cold is a mild viral infection.",
            "causes": "Rhinovirus.",
            "symptoms": "Cough, runny nose, sore throat.",
            "precautions": "Stay warm, rest."
        },
        "Migraine": {
            "about": "Migraine is a neurological condition.",
            "causes": "Triggers like stress, hormones.",
            "symptoms": "Headache, nausea, dizziness.",
            "precautions": "Avoid triggers, proper sleep."
        },
        "Diabetes": {
            "about": "Diabetes affects blood sugar levels.",
            "causes": "Insulin imbalance.",
            "symptoms": "Weight loss, fatigue, blurred vision.",
            "precautions": "Healthy diet, exercise."
        },
        "Hypertension": {
            "about": "Hypertension is high blood pressure.",
            "causes": "Lifestyle, genetics.",
            "symptoms": "Headache, dizziness.",
            "precautions": "Reduce salt, regular checkups."
        },
        "Heart Disease": {
            "about": "Heart disease affects heart function.",
            "causes": "Blocked arteries.",
            "symptoms": "Chest pain, breathlessness.",
            "precautions": "Healthy lifestyle."
        },
        "Food Poisoning": {
            "about": "Food poisoning is caused by contaminated food.",
            "causes": "Bacteria or toxins.",
            "symptoms": "Vomiting, diarrhea.",
            "precautions": "Eat clean, avoid stale food."
        }
    }

    return jsonify(explanations.get(disease, {
        "about": "No data available",
        "causes": "",
        "symptoms": "",
        "precautions": ""
    }))

# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)