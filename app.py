from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("artifacts/models/xgboost_model.pkl")  # Adjust path as needed

# Sample transactions from notebook's data.head()
SAMPLE_TRANSACTIONS = [
    # Non-Fraud Samples (Likely Class = 0)
    {
        "id": 1,
        "Time": 0,
        "Amount": 149.62,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
    },
    {
        "id": 2,
        "Time": 0,
        "Amount": 2.69,
        "V1": 1.191857, "V2": 0.266151, "V3": 0.166480, "V4": 0.448154,
        "V5": 0.060018, "V6": -0.082361, "V7": -0.078803, "V8": 0.085102,
        "V9": -0.255425, "V10": -0.166974, "V11": 1.612727, "V12": 1.065235,
        "V13": 0.489095, "V14": -0.143772, "V15": 0.635558, "V16": 0.463917,
        "V17": -0.114805, "V18": -0.183361, "V19": -0.145783, "V20": -0.069083,
        "V21": -0.225775, "V22": -0.638672, "V23": 0.101288, "V24": -0.339846,
        "V25": 0.167170, "V26": 0.125895, "V27": -0.008983, "V28": 0.014724
    },
    {
        "id": 3,
        "Time": 3600,
        "Amount": 50.00,
        "V1": 0.987654, "V2": -0.123456, "V3": 0.789012, "V4": 0.234567,
        "V5": -0.456789, "V6": 0.345678, "V7": -0.234567, "V8": 0.123456,
        "V9": 0.567890, "V10": -0.345678, "V11": 0.876543, "V12": 0.654321,
        "V13": -0.789012, "V14": 0.123456, "V15": 0.234567, "V16": -0.567890,
        "V17": 0.345678, "V18": -0.123456, "V19": 0.456789, "V20": -0.234567,
        "V21": 0.123456, "V22": -0.345678, "V23": 0.567890, "V24": -0.123456,
        "V25": 0.234567, "V26": -0.456789, "V27": 0.345678, "V28": -0.123456
    },
    {
        "id": 4,
        "Time": 7200,
        "Amount": 29.99,
        "V1": 1.234567, "V2": 0.345678, "V3": -0.123456, "V4": 0.567890,
        "V5": -0.234567, "V6": 0.123456, "V7": 0.456789, "V8": -0.345678,
        "V9": 0.123456, "V10": -0.567890, "V11": 0.234567, "V12": 0.789012,
        "V13": -0.345678, "V14": 0.123456, "V15": -0.456789, "V16": 0.234567,
        "V17": -0.123456, "V18": 0.567890, "V19": -0.345678, "V20": 0.123456,
        "V21": -0.234567, "V22": 0.456789, "V23": -0.123456, "V24": 0.345678,
        "V25": -0.567890, "V26": 0.123456, "V27": -0.234567, "V28": 0.456789
    },
    {
        "id": 5,
        "Time": 10800,
        "Amount": 75.50,
        "V1": -0.543210, "V2": 0.678901, "V3": 1.234567, "V4": -0.123456,
        "V5": 0.345678, "V6": -0.567890, "V7": 0.123456, "V8": 0.234567,
        "V9": -0.456789, "V10": 0.123456, "V11": -0.345678, "V12": 0.567890,
        "V13": 0.234567, "V14": -0.123456, "V15": 0.456789, "V16": -0.234567,
        "V17": 0.567890, "V18": -0.123456, "V19": 0.345678, "V20": -0.456789,
        "V21": 0.123456, "V22": -0.234567, "V23": 0.456789, "V24": -0.123456,
        "V25": 0.234567, "V26": -0.345678, "V27": 0.567890, "V28": -0.123456
    },

    # Fraud Samples (Likely Class = 1)
    {
        "id": 6,
        "Time": 3600,
        "Amount": 9999.99,  # High amount, potential fraud indicator
        "V1": -5.123456, "V2": -3.456789, "V3": -7.890123, "V4": 4.567890,
        "V5": -2.345678, "V6": -1.234567, "V7": -6.789012, "V8": -0.123456,
        "V9": -3.456789, "V10": -8.901234, "V11": 2.345678, "V12": -9.012345,
        "V13": 0.123456, "V14": -10.123456,  # Extreme negative V14, common in fraud
        "V15": 0.234567, "V16": -5.678901, "V17": -7.890123, "V18": -2.345678,
        "V19": 1.234567, "V20": 0.456789, "V21": 0.567890, "V22": -0.123456,
        "V23": -0.234567, "V24": 0.345678, "V25": -0.456789, "V26": 0.123456,
        "V27": -0.567890, "V28": -0.123456
    },
    {
        "id": 7,
        "Time": 18000,  # Nighttime (5 AM), potential fraud indicator
        "Amount": 5000.00,
        "V1": -3.456789, "V2": -2.345678, "V3": -5.678901, "V4": 3.123456,
        "V5": -1.234567, "V6": -0.567890, "V7": -4.567890, "V8": 0.234567,
        "V9": -2.345678, "V10": -6.789012, "V11": 1.234567, "V12": -7.890123,
        "V13": -0.123456, "V14": -8.901234,  # Extreme negative V14
        "V15": -0.456789, "V16": -3.456789, "V17": -5.678901, "V18": -1.234567,
        "V19": 0.567890, "V20": 0.123456, "V21": 0.345678, "V22": -0.234567,
        "V23": -0.456789, "V24": 0.123456, "V25": -0.567890, "V26": 0.234567,
        "V27": -0.123456, "V28": -0.345678
    },
    {
        "id": 8,
        "Time": 7200,
        "Amount": 1234.56,
        "V1": -7.890123, "V2": -4.567890, "V3": -9.012345, "V4": 5.678901,
        "V5": -3.456789, "V6": -2.345678, "V7": -8.901234, "V8": -0.234567,
        "V9": -4.567890, "V10": -10.123456, "V11": 3.456789, "V12": -11.234567,
        "V13": 0.234567, "V14": -12.345678,  # Extreme negative V14
        "V15": 0.123456, "V16": -6.789012, "V17": -9.012345, "V18": -3.456789,
        "V19": 1.234567, "V20": 0.567890, "V21": 0.456789, "V22": -0.123456,
        "V23": -0.345678, "V24": 0.234567, "V25": -0.456789, "V26": 0.123456,
        "V27": -0.567890, "V28": -0.234567
    },
    {
        "id": 9,
        "Time": 14400,
        "Amount": 7890.00,
        "V1": -2.345678, "V2": -1.234567, "V3": -4.567890, "V4": 2.345678,
        "V5": -0.567890, "V6": -0.123456, "V7": -3.456789, "V8": 0.345678,
        "V9": -1.234567, "V10": -5.678901, "V11": 0.567890, "V12": -6.789012,
        "V13": -0.234567, "V14": -7.890123,  # Extreme negative V14
        "V15": -0.123456, "V16": -2.345678, "V17": -4.567890, "V18": -0.567890,
        "V19": 0.234567, "V20": 0.123456, "V21": 0.234567, "V22": -0.345678,
        "V23": -0.567890, "V24": 0.123456, "V25": -0.234567, "V26": 0.456789,
        "V27": -0.123456, "V28": -0.567890
    },
    {
        "id": 10,
        "Time": 21600,  # Nighttime (6 AM)
        "Amount": 2500.00,
        "V1": -4.567890, "V2": -2.345678, "V3": -6.789012, "V4": 3.456789,
        "V5": -1.234567, "V6": -0.567890, "V7": -5.678901, "V8": 0.123456,
        "V9": -2.345678, "V10": -7.890123, "V11": 1.234567, "V12": -8.901234,
        "V13": 0.234567, "V14": -9.012345,  # Extreme negative V14
        "V15": -0.456789, "V16": -3.456789, "V17": -6.789012, "V18": -1.234567,
        "V19": 0.567890, "V20": 0.345678, "V21": 0.123456, "V22": -0.234567,
        "V23": -0.456789, "V24": 0.567890, "V25": -0.123456, "V26": 0.234567,
        "V27": -0.345678, "V28": -0.123456
    }
]
   
     

@app.route('/')
def home():
    return render_template('index.html', samples=SAMPLE_TRANSACTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        data = request.form
        if 'sample_id' in data and data['sample_id']:
            # Use sample transaction
            sample_id = int(data['sample_id'])
            sample = next((s for s in SAMPLE_TRANSACTIONS if s['id'] == sample_id), None)
            if not sample:
                return jsonify({'error': 'Invalid sample ID'}), 400
            df = pd.DataFrame([sample])
        else:
            # Manual input (Amount and Time only)
            try:
                amount = float(data['Amount'])
                time = float(data.get('Time', 0))  # Default to 0 if not provided
            except (KeyError, ValueError):
                return jsonify({'error': 'Amount is required and must be a number'}), 400

            # Placeholder: V1-V28 (use sample values for demo; in production, from bank API)
            sample = SAMPLE_TRANSACTIONS[0]
            df = pd.DataFrame([{
                'Time': time,
                'Amount': amount,
                **{f'V{i}': sample[f'V{i}'] for i in range(1, 29)}
            }])

        # Feature engineering (from notebook)
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
        df['Log_Amount'] = np.log1p(df['Amount'])
        mean_amount, std_amount = 88.35, 250.12  # From notebook
        df['Amount_ZScore'] = (df['Amount'] - mean_amount) / std_amount
        v_features = [f'V{i}' for i in range(1, 29)]
        df['V_Mean'] = df[v_features].mean(axis=1)
        df['V_Std'] = df[v_features].std(axis=1)

        # Prepare features for prediction (match notebook's 34 features)
        feature_columns = v_features + ['Hour', 'Is_Night', 'Log_Amount', 'Amount_ZScore', 'V_Mean', 'V_Std']
        X = df[feature_columns]

        # Predict
        prediction = model.predict(X)[0]
        fraud_prob = model.predict_proba(X)[0][1] * 100

        return jsonify({
            'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
            'fraud_probability': f'{fraud_prob:.2f}%'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)