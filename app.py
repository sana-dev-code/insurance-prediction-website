from flask import Flask, request, redirect, url_for, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ------------------------ Dashboard Page ------------------------
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Predictor</title>
    <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            padding: 50px;
        }
        h1 {
            font-size: 3rem;
        }
        p {
            font-size: 1.2rem;
            margin-top: 20px;
        }
        img {
            max-width: 400px;
            border-radius: 15px;
            margin-top: 30px;
        }
        a {
            display: inline-block;
            margin-top: 30px;
            padding: 15px 30px;
            background: white;
            color: #2575fc;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Insurance Charges Predictor</h1>
    <p>This project estimates medical insurance charges based on your details like age, gender, BMI, smoking status, and region.</p>
    <img src="https://cdn-icons-png.flaticon.com/512/2332/2332400.png" alt="Insurance Image">
    <br>
    <a href="{{ url_for('predict') }}">Start Prediction</a>
</body>
</html>
"""

# ------------------------ Prediction Page ------------------------
predict_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Enter Details</title>
    <style>
        body {
            background: #f0f2f5;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .form-box {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            width: 400px;
        }
        h2 {
            text-align: center;
            margin-bottom: 30px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #2575fc;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            text-align: center;
            font-size: 1.2rem;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="form-box">
        <h2>Insurance Charge Estimator</h2>
        <form method="POST">
            <input type="number" name="age" placeholder="Age" required>
            <select name="sex" required>
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
            <input type="number" step="any" name="bmi" placeholder="BMI" required>
            <input type="number" name="children" placeholder="Number of Children" required>
            <select name="smoker" required>
                <option value="">Smoker?</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
            </select>
            <select name="region" required>
                <option value="">Select Region</option>
                <option value="northeast">Northeast</option>
                <option value="northwest">Northwest</option>
                <option value="southeast">Southeast</option>
                <option value="southwest">Southwest</option>
            </select>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
        <div class="result">
            <p><strong>Predicted Insurance Charges:</strong> ${{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ------------------------ Flask Routes ------------------------
@app.route('/')
def dashboard():
    return render_template_string(dashboard_html)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = request.form['sex']
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']

            input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                    columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            prediction = round(model.predict(input_df)[0], 2)
        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template_string(predict_html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
