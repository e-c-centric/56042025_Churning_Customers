from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model1 = load_model('optimized_model1.h5')
model2 = load_model('optimized_model2.h5')
unoptimized_model = load_model('unoptimized_model.h5')

# Assigning weights to each model
weights = {
    'unoptimized_model': 2,
    'optimized_model1': 1,
    'optimized_model2': 1
}

scaler = StandardScaler()
training_data = pd.read_csv('final_data.csv')
scaler.fit(training_data.drop(columns=['Churn'], axis=1))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Dependents': int(request.form['Dependents']),
            'tenure': int(request.form['tenure']),
            'OnlineSecurity': int(request.form['OnlineSecurity']),
            'OnlineBackup': int(request.form['OnlineBackup']),
            'DeviceProtection': int(request.form['DeviceProtection']),
            'TechSupport': int(request.form['TechSupport']),
            'Contract': int(request.form['Contract']),
            'PaperlessBilling': int(request.form['PaperlessBilling']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])  # Added TotalCharges
        }

        input_df = pd.DataFrame([input_data])

        input_scaled = scaler.transform(input_df)

        prediction_unoptimized = unoptimized_model.predict(input_scaled)
        prediction_optimized1 = model1.predict(input_scaled)
        prediction_optimized2 = model2.predict(input_scaled)

        consensus_prediction = (weights['unoptimized_model'] * prediction_unoptimized +
                                weights['optimized_model1'] * prediction_optimized1 +
                                weights['optimized_model2'] * prediction_optimized2) / sum(weights.values())

        result = "Churn" if consensus_prediction[0, 0] > 0.5 else "No Churn"

        confidence = 1 - 2 * abs(consensus_prediction[0, 0] - 0.5)**2
        confidence = round(confidence * 100, 2)
        return render_template('results.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)