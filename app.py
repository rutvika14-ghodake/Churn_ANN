```python
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ============================================================
# LOAD TRAINED ANN MODEL
# ============================================================

MODEL_PATH = "modelANN.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ============================================================
# HTML + CSS
# ============================================================

HTML_TEMPLATE = """

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Customer Churn Predictor</title>

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
          rel="stylesheet">

    <style>

        :root {

            --bg-gradient:
                linear-gradient(135deg, #0f172a 0%, #1e293b 100%);

            --card-bg: rgba(30, 41, 59, 0.75);

            --accent-color: #06b6d4;

            --accent-hover: #0891b2;

            --text-main: #f8fafc;

            --text-muted: #94a3b8;

            --input-bg: #0f172a;

            --input-border: #334155;

            --success-color: #10b981;

            --danger-color: #ef4444;

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

            border: 1px solid rgba(255,255,255,0.1);

            border-radius: 18px;

            padding: 2.5rem;

            width: 100%;

            max-width: 800px;

            box-shadow:
                0 20px 25px -5px rgba(0,0,0,0.5);

        }


        .header {

            text-align: center;

            margin-bottom: 2rem;

        }


        .header h1 {

            font-size: 2.2rem;

            font-weight: 700;

            background:
                linear-gradient(to right, #38bdf8, #06b6d4);

            -webkit-background-clip: text;

            -webkit-text-fill-color: transparent;

            margin-bottom: 0.6rem;

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

            font-size: 0.82rem;

            font-weight: 600;

            color: var(--text-muted);

            margin-bottom: 0.45rem;

            text-transform: uppercase;

            letter-spacing: 0.05em;

        }


        .form-group input,

        .form-group select {

            width: 100%;

            background: var(--input-bg);

            border: 1px solid var(--input-border);

            color: var(--text-main);

            padding: 0.8rem 1rem;

            border-radius: 9px;

            font-size: 0.95rem;

            outline: none;

            transition: all 0.2s ease;

        }


        .form-group input:focus,

        .form-group select:focus {

            border-color: var(--accent-color);

            box-shadow:
                0 0 0 3px rgba(6,182,212,0.2);

        }


        .form-group select option {

            background: #0f172a;

            color: white;

        }


        .btn-submit {

            grid-column: 1 / -1;

            margin-top: 0.8rem;

            background: var(--accent-color);

            color: #0f172a;

            font-weight: 700;

            font-size: 1rem;

            padding: 0.9rem;

            border: none;

            border-radius: 9px;

            cursor: pointer;

            transition: all 0.2s ease;

        }


        .btn-submit:hover {

            background: var(--accent-hover);

            transform: translateY(-1px);

        }


        .btn-submit:active {

            transform: scale(0.99);

        }


        .result-box {

            margin-top: 2rem;

            padding: 1.5rem;

            border-radius: 10px;

            text-align: center;

        }


        .result-box.stay {

            background: rgba(16,185,129,0.1);

            border: 1px solid rgba(16,185,129,0.35);

        }


        .result-box.exit {

            background: rgba(239,68,68,0.1);

            border: 1px solid rgba(239,68,68,0.35);

        }


        .result-box h2 {

            font-size: 1rem;

            color: var(--text-muted);

            margin-bottom: 0.4rem;

        }


        .prediction-val {

            font-size: 1.6rem;

            font-weight: 700;

        }


        .stay .prediction-val {

            color: var(--success-color);

        }


        .exit .prediction-val {

            color: var(--danger-color);

        }


        .error {

            margin-top: 1.5rem;

            padding: 1rem;

            border-radius: 8px;

            background: rgba(239,68,68,0.1);

            border: 1px solid rgba(239,68,68,0.3);

            color: #fca5a5;

            text-align: center;

        }

    </style>

</head>


<body>


<div class="container">


    <div class="header">

        <h1>Customer Churn Predictor</h1>

        <p>
            Enter customer details to predict whether the customer
            is likely to stay or exit.
        </p>

    </div>


    <form action="/predict"
          method="POST"
          class="grid-form">


        <!-- ================================================= -->
        <!-- CREDIT SCORE -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="creditScore">
                Credit Score
            </label>

            <input
                type="number"
                id="creditScore"
                name="creditScore"
                placeholder="e.g. 619"
                value="{{ values.get('creditScore', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- GEOGRAPHY -->
        <!-- LabelEncoder: France=0, Germany=1, Spain=2 -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="geography">
                Geography
            </label>

            <select
                id="geography"
                name="geography"
                required
            >

                <option value="">
                    Select Country
                </option>

                <option value="0"
                    {% if values.get('geography') == '0' %}
                    selected
                    {% endif %}>
                    France
                </option>

                <option value="1"
                    {% if values.get('geography') == '1' %}
                    selected
                    {% endif %}>
                    Germany
                </option>

                <option value="2"
                    {% if values.get('geography') == '2' %}
                    selected
                    {% endif %}>
                    Spain
                </option>

            </select>

        </div>


        <!-- ================================================= -->
        <!-- GENDER -->
        <!-- LabelEncoder: Female=0, Male=1 -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="gender">
                Gender
            </label>

            <select
                id="gender"
                name="gender"
                required
            >

                <option value="">
                    Select Gender
                </option>

                <option value="0"
                    {% if values.get('gender') == '0' %}
                    selected
                    {% endif %}>
                    Female
                </option>

                <option value="1"
                    {% if values.get('gender') == '1' %}
                    selected
                    {% endif %}>
                    Male
                </option>

            </select>

        </div>


        <!-- ================================================= -->
        <!-- AGE -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="age">
                Age
            </label>

            <input
                type="number"
                id="age"
                name="age"
                placeholder="e.g. 42"
                value="{{ values.get('age', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- TENURE -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="tenure">
                Tenure
            </label>

            <input
                type="number"
                id="tenure"
                name="tenure"
                min="0"
                max="10"
                placeholder="e.g. 2"
                value="{{ values.get('tenure', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- BALANCE -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="balance">
                Balance
            </label>

            <input
                type="number"
                step="any"
                id="balance"
                name="balance"
                placeholder="e.g. 83807.86"
                value="{{ values.get('balance', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- NUMBER OF PRODUCTS -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="numOfProducts">
                Number of Products
            </label>

            <input
                type="number"
                id="numOfProducts"
                name="numOfProducts"
                min="1"
                max="4"
                placeholder="e.g. 1"
                value="{{ values.get('numOfProducts', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- HAS CREDIT CARD -->
        <!-- LabelEncoder-style binary value: Yes=1, No=0 -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="hasCrCard">
                Has Credit Card
            </label>

            <select
                id="hasCrCard"
                name="hasCrCard"
                required
            >

                <option value="">
                    Select
                </option>

                <option value="1"
                    {% if values.get('hasCrCard') == '1' %}
                    selected
                    {% endif %}>
                    Yes
                </option>

                <option value="0"
                    {% if values.get('hasCrCard') == '0' %}
                    selected
                    {% endif %}>
                    No
                </option>

            </select>

        </div>


        <!-- ================================================= -->
        <!-- ACTIVE MEMBER -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="isActiveMember">
                Is Active Member
            </label>

            <select
                id="isActiveMember"
                name="isActiveMember"
                required
            >

                <option value="">
                    Select
                </option>

                <option value="1"
                    {% if values.get('isActiveMember') == '1' %}
                    selected
                    {% endif %}>
                    Yes
                </option>

                <option value="0"
                    {% if values.get('isActiveMember') == '0' %}
                    selected
                    {% endif %}>
                    No
                </option>

            </select>

        </div>


        <!-- ================================================= -->
        <!-- ESTIMATED SALARY -->
        <!-- ================================================= -->

        <div class="form-group">

            <label for="estimatedSalary">
                Estimated Salary
            </label>

            <input
                type="number"
                step="any"
                id="estimatedSalary"
                name="estimatedSalary"
                placeholder="e.g. 101348.88"
                value="{{ values.get('estimatedSalary', '') }}"
                required
            >

        </div>


        <!-- ================================================= -->
        <!-- SUBMIT -->
        <!-- ================================================= -->

        <button
            type="submit"
            class="btn-submit"
        >
            Run Prediction
        </button>


    </form>


    <!-- ===================================================== -->
    <!-- RESULT -->
    <!-- ===================================================== -->

    {% if prediction_text is not none %}

        {% if prediction_class == "stay" %}

            <div class="result-box stay">

                <h2>Prediction Result</h2>

                <div class="prediction-val">
                    {{ prediction_text }}
                </div>

            </div>

        {% elif prediction_class == "exit" %}

            <div class="result-box exit">

                <h2>Prediction Result</h2>

                <div class="prediction-val">
                    {{ prediction_text }}
                </div>

            </div>

        {% else %}

            <div class="error">
                {{ prediction_text }}
            </div>

        {% endif %}

    {% endif %}


</div>


</body>

</html>

"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=None,
        prediction_class=None,
        values={}
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="Error: Model could not be loaded.",
            prediction_class="error",
            values=request.form
        )


    try:

        # ====================================================
        # GET FORM VALUES
        # ====================================================

        credit_score = float(
            request.form.get("creditScore")
        )

        geography = float(
            request.form.get("geography")
        )

        gender = float(
            request.form.get("gender")
        )

        age = float(
            request.form.get("age")
        )

        tenure = float(
            request.form.get("tenure")
        )

        balance = float(
            request.form.get("balance")
        )

        num_products = float(
            request.form.get("numOfProducts")
        )

        has_cr_card = float(
            request.form.get("hasCrCard")
        )

        is_active_member = float(
            request.form.get("isActiveMember")
        )

        estimated_salary = float(
            request.form.get("estimatedSalary")
        )


        # ====================================================
        # CREATE INPUT ARRAY
        #
        # EXACT SAME ORDER AS TRAINING DATA
        # ====================================================

        final_features = np.array([[
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_products,
            has_cr_card,
            is_active_member,
            estimated_salary
        ]])


        print("Input received:")
        print(final_features)


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        raw_pred = model.predict(final_features)

        pred_value = float(raw_pred[0][0])


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if pred_value >= 0.5:

            class_label = 1

            result = (
                f"Customer Likely to EXIT "
                f"(Probability: {pred_value:.2%})"
            )

            prediction_class = "exit"

        else:

            class_label = 0

            result = (
                f"Customer Likely to STAY "
                f"(Probability: {(1 - pred_value):.2%})"
            )

            prediction_class = "stay"


        print(
            f"Prediction: {class_label}"
        )

        print(
            f"Probability: {pred_value:.4f}"
        )


        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=result,
            prediction_class=prediction_class,
            values=request.form
        )


    except Exception as e:

        print(f"Prediction error: {e}")

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"Prediction Error: {str(e)}",
            prediction_class="error",
            values=request.form
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
```
