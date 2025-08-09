from flask import Flask, request, render_template_string
import joblib
import pandas as pd

# Daftar fitur sesuai urutan model
FEATURES = [
    'Age',
    'Chest Pain',
    'Persistent Cough',
    'Snoring/Sleep Apnea',
    'Excessive Sweating',
    'Cold Hands/Feet',
    'Shortness of Breath',
    'Chest Discomfort (Activity)',
    'Pain in Neck/Jaw/Shoulder/Back',
    'Nausea/Vomiting',
    'Fatigue & Weakness',
    'Anxiety/Feeling of Doom',
    'Swelling (Edema)',
    'High Blood Pressure',
    'Irregular Heartbeat',
    'Dizziness'
]

# Load model pipeline
model = joblib.load("C:\\.kampus\\DE\\Uas mpml\\svm_stroke_model.pkl")

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Stroke Risk Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f7fa; padding: 40px; }
        .container { max-width: 500px; margin: 0 auto; }
        .card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1);}
        h2 { text-align: center; }
        .form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .form-group { margin-bottom: 10px; }
        label { display: block; margin-bottom: 5px; }
        select, input { width: 100%; padding: 8px; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #3498db; color: white; padding: 10px; border: none; border-radius: 8px; width: 100%; margin-top: 10px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; }
        .risk { background: #e74c3c; color: white; }
        .safe { background: #2ecc71; color: white; }
        .error { background: #f39c12; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Stroke Risk Prediction</h2>
        <div class="card">
            <form method="POST">
                <div class="form-grid">
                    {% for feature in features %}
                    <div class="form-group">
                        <label>{{ feature }}:</label>
                        {% if feature == 'Age' %}
                        <input type="number" name="Age" min="1" max="120" required>
                        {% else %}
                        <select name="{{ feature }}" required>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                <button type="submit">Predict Risk</button>
            </form>
            {% if prediction is not none %}
                <div class="result {% if prediction == 'At Risk' %}risk{% elif prediction.startswith('Error') %}error{% else %}safe{% endif %}">
                    {{ prediction }}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            input_data = {}
            for feature in FEATURES:
                value = request.form.get(feature)
                input_data[feature] = int(value)
            df = pd.DataFrame([input_data])[FEATURES]
            result = model.predict(df)[0]
            prediction = "At Risk" if result == 1 else "Not at Risk"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(HTML_FORM, features=FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
