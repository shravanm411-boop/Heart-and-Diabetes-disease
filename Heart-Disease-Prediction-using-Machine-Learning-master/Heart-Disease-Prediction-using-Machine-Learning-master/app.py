from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------
# Load all required models
# -----------------------------
heart_model = joblib.load('heart_disease_model_rf.pkl')
diabetes_model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')   # Used for diabetes input
pca = joblib.load('pca.pkl')         # Used for dimensionality reduction in diabetes

# -----------------------------
# Landing Page (Welcome)
# -----------------------------
@app.route('/')
def welcome():
    return render_template('welcome.html')

# -----------------------------
# Homepage (index with choices)
# -----------------------------
@app.route('/home')
def home():
    return render_template('index.html')

# -----------------------------
# Heart Disease Prediction
# -----------------------------
@app.route('/predict_heart')
def heart_form():
    return render_template('heart_form.html')

@app.route('/result_heart', methods=['POST'])
def result_heart():
    try:
        # Collect input features from form
        features = [float(request.form[key]) for key in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        # Predict using heart disease model
        prediction = heart_model.predict([features])[0]
        result = "at risk of Heart Disease" if prediction == 1 else "not at risk of Heart Disease"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('heart_form.html', error="Invalid input: " + str(e))

# -----------------------------
# Diabetes Prediction
# -----------------------------
@app.route('/predict_diabetes')
def diabetes_form():
    return render_template('diabetes_form.html')

@app.route('/result_diabetes', methods=['POST'])
def result_diabetes():
    try:
        # Collect diabetes input features
        features = [float(request.form[key]) for key in [
            'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
            'insulin', 'bmi', 'dpf', 'age'
        ]]
        # Apply standardization and PCA transformation
        scaled = scaler.transform([features])
        reduced = pca.transform(scaled)
        prediction = diabetes_model.predict(reduced)[0]
        result = "likely Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('diabetes_form.html', error="Invalid input: " + str(e))

# -----------------------------
# Combined Prediction (Heart + Diabetes)
# -----------------------------
@app.route('/combined_form')
def combined_form():
    return render_template('combined_form.html')

@app.route('/result_combined', methods=['POST'])
def result_combined():
    try:
        # --- Heart disease prediction ---
        heart_features = [float(request.form[key]) for key in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        heart_pred = heart_model.predict([heart_features])[0]
        heart_result = "at risk of Heart Disease" if heart_pred == 1 else "not at risk of Heart Disease"

        # --- Diabetes prediction ---
        diabetes_features = [float(request.form[key]) for key in [
            'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
            'insulin', 'bmi', 'dpf', 'age'
        ]]
        scaled = scaler.transform([diabetes_features])
        reduced = pca.transform(scaled)
        diabetes_pred = diabetes_model.predict(reduced)[0]
        diabetes_result = "likely Diabetic" if diabetes_pred == 1 else "Not Diabetic"

        # ✅ Pass correct variable names for template
        return render_template(
            'result_combined.html',
            heart_result=heart_result,
            diabetes_result=diabetes_result
        )
    except Exception as e:
        return render_template('combined_form.html', error="Invalid input: " + str(e))

# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
