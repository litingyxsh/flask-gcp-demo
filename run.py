from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)


model = pickle.load(open("rf_model.pkl", "rb"))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        age = float(request.form['person_age'])
        income = float(request.form['person_income'])
        emp_length = float(request.form['person_emp_length'])
        loan_amnt = float(request.form['loan_amnt'])
        int_rate = float(request.form['loan_int_rate'])
        percent_income = float(request.form['loan_percent_income'])
        cred_hist = float(request.form['cb_person_cred_hist_length'])

        home_mortgage = int(request.form['person_home_ownership_MORTGAGE'])
        home_other = int(request.form['person_home_ownership_OTHER'])
        home_own = int(request.form['person_home_ownership_OWN'])
        home_rent = int(request.form['person_home_ownership_RENT'])

        intent_debt = int(request.form['loan_intent_DEBTCONSOLIDATION'])
        intent_edu = int(request.form['loan_intent_EDUCATION'])
        intent_home = int(request.form['loan_intent_HOMEIMPROVEMENT'])
        intent_med = int(request.form['loan_intent_MEDICAL'])
        intent_personal = int(request.form['loan_intent_PERSONAL'])
        intent_venture = int(request.form['loan_intent_VENTURE'])

        loan_grade = int(request.form['loan_grade_enc'])
        cb_default = int(request.form['cb_person_default_on_file_enc'])

        
        features = np.array([[age, income, emp_length, loan_amnt, int_rate, percent_income, cred_hist,
                              home_mortgage, home_other, home_own, home_rent,
                              intent_debt, intent_edu, intent_home, intent_med, intent_personal, intent_venture,
                              loan_grade, cb_default]])

        
        prediction = model.predict(features)[0]
        result = "Likely to Default" if prediction == 1 else "Not Likely to Default"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
