import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Keras ANN model saved as modelANN.pkl
MODEL_PATH = "modelANN.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single-file HTML template with CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Model Deployment</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #06b6d4;
            --accent-hover: #0891b2;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: #0f172a;
            --input-border: #334155;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.5rem;
            width: 100%;
            max-width: 680px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 580px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group input {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-group input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-color);
            color: #0f172a;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.85rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            background: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 8px;
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .result-box .prediction-val {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--success-color);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>ANN Model Predictor</h1>
        <p>Enter the 10 numerical features below to generate a prediction.</p>
    </div>

    <form action="/predict" method="POST" class="grid-form">
        {% for i in range(1, 11) %}
        <div class="form-group">
            <label for="f{{ i }}">Feature {{ i }}</label>
            <input 
                type="number" 
                step="any" 
                id="f{{ i }}" 
                name="feature{{ i }}" 
                placeholder="0.0" 
                value="{{ inputs[i-1] if inputs else '' }}" 
                required>
        </div>
        {% endfor %}

        <button type="submit" class="btn-submit">Run Prediction</button>
    </form>

    {% if prediction_text is not none %}
    <div class="result-box">
        <h2>Output Result</h2>
        <div class="prediction-val">{{ prediction_text }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None, inputs=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="Error: Model not loaded.",
            inputs=None,
        )

    try:
        # Extract inputs for the 10 features
        raw_inputs = [request.form.get(f"feature{i}") for i in range(1, 11)]
        float_features = [float(x) for x in raw_inputs]

        # Reshape input to (1, 10) for model inference
        final_features = np.array([float_features])

        # Get raw prediction score (Sigmoid output: 0 to 1)
        raw_pred = model.predict(final_features)
        pred_value = float(raw_pred[0][0])

        # Apply 0.5 classification threshold
        class_label = 1 if pred_value >= 0.5 else 0
        result = f"Class {class_label} (Probability: {pred_value:.4f})"

        return render_template_string(
            HTML_TEMPLATE, prediction_text=result, inputs=raw_inputs
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, prediction_text=f"Prediction Error: {str(e)}", inputs=None
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
