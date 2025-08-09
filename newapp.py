from streamlit.components.v1 import html
import streamlit as st
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
<form id="stroke-form">
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:15px;">
        <div>
            <label>Age:</label>
            <input type="number" name="Age" min="1" max="120" required>
        </div>
        {fields}
    </div>
    <button type="submit" style="margin-top:20px;background:#3498db;color:white;padding:10px;border:none;border-radius:8px;width:100%;">Predict Risk</button>
</form>
<div id="result" style="margin-top:20px;font-weight:bold;font-size:18px;"></div>
<script>
const form = document.getElementById('stroke-form');
form.onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(form);
    let data = {};
    for (let [k,v] of formData.entries()) data[k]=v;
    const resp = await fetch(window.location.pathname, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(data)
    });
    const result = await resp.json();
    document.getElementById('result').innerHTML = result.prediction;
    document.getElementById('result').style.background = result.prediction === "At Risk" ? "#e74c3c" : "#2ecc71";
    document.getElementById('result').style.color = "white";
    document.getElementById('result').style.padding = "15px";
    document.getElementById('result').style.borderRadius = "8px";
};
</script>
"""

def make_fields():
    html_fields = ""
    for feat in FEATURES[1:]:
        html_fields += f"""
        <div>
            <label>{feat}:</label>
            <select name="{feat}" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        """
    return html_fields

st.set_page_config(page_title="Stroke Risk Prediction", layout="centered")
st.title("🧠 Stroke Risk Prediction")
st.write("Isi form berikut untuk memprediksi risiko stroke:")

fields_html = make_fields()
form_html = HTML_FORM.format(fields=fields_html)

def predict_api():
    import streamlit.web.server.websocket_headers
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.runtime import get_instance
    from streamlit.web.server import Server
    import json

    # Streamlit's hacky way to get the request body
    ctx = get_script_run_ctx()
    if ctx is not None and hasattr(ctx, "request") and ctx.request is not None:
        try:
            body = ctx.request.body.decode()
            data = json.loads(body)
            input_data = {}
            for feat in FEATURES:
                input_data[feat] = int(data.get(feat, 0))
            df = pd.DataFrame([input_data])[FEATURES]
            result = model.predict(df)[0]
            prediction = "At Risk" if result == 1 else "Not at Risk"
            return {"prediction": prediction}
        except Exception as e:
            return {"prediction": f"Error: {str(e)}"}
    return None

if st._is_running_with_streamlit:
    # Show HTML form
    html(form_html, height=600)

    # Handle POST (AJAX) requests
    import streamlit.web.server.websocket_headers
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.runtime import get_instance
    from streamlit.web.server import Server
    import json

    # Monkey patching to add a POST endpoint (for demo, not production)
    if not hasattr(st, "_custom_api_registered"):
        from streamlit.web.server import routes
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @routes.app.post("/")
        async def predict_route(request: Request):
            data = await request.json()
            input_data = {}
            for feat in FEATURES:
                input_data[feat] = int(data.get(feat, 0))
            df = pd.DataFrame([input_data])[FEATURES]
            try:
                result = model.predict(df)[0]
                prediction = "At Risk" if result == 1 else "Not at Risk"
            except Exception as e:
                prediction = f"Error: {str(e)}"
            return JSONResponse({"prediction": prediction})

        st._custom_api_registered = True
