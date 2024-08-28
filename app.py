from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle
import pandas as pd
import warnings
import lightgbm as lgb


warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

# Elias Alkharma
blood_glucose_model = joblib.load('BloodGlucose_model.pkl')
blood_glucose_scaler = joblib.load('scalerBloodGlucose.pkl')
blood_glucose_selector = joblib.load('selectorBloodGlucose.pkl')



# Nayaz
heart_disease_model = pickle.load(open('heart_disease_pred.sav', 'rb'))

# Amaar
with open('Parkinsson_model.pkl', 'rb') as file:
    parkinsson_model = pickle.load(file)

# Ayman
with open('thyroid_disease_detection.sav', 'rb') as file:
    thyroid_model = joblib.load(file)

# Maha
Hepatitis_model = joblib.load('Hepatitis_model.pkl')
Hepatitis_scaler = joblib.load('Hepatitis_scalar.pkl')
Hypertension_model = joblib.load('Hypertension_model.pkl')

# Hanin
Asthma_model = joblib.load('Asthma_model.pkl')


# Load the LightGBM model
Deepression_model = joblib.load('Deepression_model.pkl')
Deepression_scaler = joblib.load('Deepression_scaler.pkl')


@app.route('/')
def index():
    return render_template('index.html')

def validate_input(data):
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    for field in required_fields:
        if field not in data:
            return False, f'Missing field: {field}'
        if not isinstance(data[field], (int, float)):
            return False, f'Invalid type for field: {field}'
    return True, ''

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        data = request.json
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        try:
            input_features = np.array([[
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['SkinThickness'],
                data['Insulin'],
                data['BMI'],
                data['DiabetesPedigreeFunction'],
                data['Age']
            ]])
            scaled_features = blood_glucose_scaler.transform(input_features)
            selected_features = blood_glucose_selector.transform(scaled_features)
            blood_glucose_prediction = blood_glucose_model.predict(selected_features)[0]

            return jsonify({
                'blood_glucose_prediction': int(blood_glucose_prediction)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page1.html')


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        data = request.json
        try:
            input_data = [
                int(data.get('Age', 0)),
                int(data.get('Sex', 0)),
                int(data.get('ChestPainType', 0)),
                int(data.get('RestingBP', 0)),
                int(data.get('Cholesterol', 0)),
                int(data.get('FastingBS', 0)),
                int(data.get('RestingECG', 0)),
                int(data.get('MaxHR', 0)),
                int(data.get('ExerciseAngina', 0)),
                float(data.get('Oldpeak', 0.0)),
                int(data.get('ST_Slope', 0))
            ]

            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            prediction = heart_disease_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                diagnosis = 'Low Risk Detected'
            else:
                diagnosis = 'High Risk Detected'

            return jsonify({'diagnosis': diagnosis})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page2.html')

@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        data = request.json
        input_data = [
            int(data.get('MDVP:Fo(Hz)', 0)),
            int(data.get('MDVP:Fhi(Hz)', 0)),
            int(data.get('MDVP:Flo(Hz)', 0)),
            float(data.get('MDVP:Jitter(%)', 0.00)),
            float(data.get('Jitter:DDP', 0.00)),
            float(data.get('Shimmer:APQ5', 0.00)),
            int(data.get('HNR', 0)),
            float(data.get('spread1', 0.00)),
            float(data.get('spread2', 0.00)),
            float(data.get('PPE', 0.00)),
        ]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = parkinsson_model.predict(input_data_reshaped)
        if prediction == 1:
            result = 'Have Parkinson'
        elif prediction == 0:
            result = 'Do Not Have Parkinson'
        else:
            result = 'An error occurred: ' + prediction

        return jsonify({'Parkinson': result})

    return render_template('page3.html')

@app.route('/page4', methods=['GET', 'POST'])
def page4():
    if request.method == 'POST':
        data = request.json
        try:
            input_features = pd.DataFrame({
                'age': [int(data.get('age', 0))],
                'sex': [int(data.get('sex', 0))],
                'TT4': [float(data.get('TT4', 0.0))],
                'T3': [float(data.get('T3', 0.0))],
                'T4U': [float(data.get('T4U', 0.0))],
                'FTI': [float(data.get('FTI', 0.0))],
                'TSH': [float(data.get('TSH', 0.0))],
                'pregnant': [int(data.get('pregnant', 0))]
            })

            input_data_as_numpy_array2 = np.asarray(input_features)
            input_data_reshaped2 = input_data_as_numpy_array2.reshape(1, -1)
            prediction = thyroid_model.predict(input_features)[0]

            if prediction == 1:
                resul = 'Hypothyroidism'
            elif prediction == 2:
                resul = 'normal condition and does not suffer from any thyroid problems.'
            else:
                resul = 'Hyperthyroidism.'

            return jsonify({'diagnosis': resul})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page4.html')

@app.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        data = request.json
        try:
            input_features = pd.DataFrame({
                'Age': [int(data.get('Age', 0))],
                'Sex': [int(data.get('Sex', 0))],
                'ALB': [float(data.get('ALB', 0.0))],
                'ALP': [float(data.get('ALP', 0.0))],
                'ALT': [float(data.get('ALT', 0.0))],
                'AST': [float(data.get('AST', 0.0))],
                'BIL': [float(data.get('BIL', 0.0))],
                'CHE': [int(data.get('CHE', 0.0))],
                'CHOL': [int(data.get('CHOL', 0.0))],
                'CREA': [int(data.get('CREA', 0.0))],
                'GGT': [int(data.get('GGT', 0.0))],
                'PROT': [int(data.get('PROT', 0.0))],
            })

            input_features = np.insert(input_features, 0, 0, axis=1)
            scaled_test_data = Hepatitis_scaler.transform(input_features)
            data = scaled_test_data[:, 1:]
            prediction = Hepatitis_model.predict(data)[0]

            return jsonify({'knn_prediction': int(prediction)})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page5.html')



@app.route('/page6', methods=['GET', 'POST'])
def page6():
    if request.method == 'POST':
        try:
            # Extract JSON data from the request
            data = request.json

            # Extract features and convert to integers
            features = [
                int(data.get('angry', 0)),
                int(data.get('fear', 0)),
                int(data.get('disgust', 0)),
                int(data.get('happy', 0)),
                int(data.get('neutral', 0)),
                int(data.get('sad', 0)),
                int(data.get('surprise', 0)),
            ]

            # Convert features to numpy array and reshape for the model
            features_array = np.array([features])

            # Scale the data
            scaled_data = Deepression_scaler.transform(features_array)

            # Make prediction
            prediction = Deepression_model.predict(scaled_data)[0]

            # Return prediction as JSON
            return jsonify({'Deepression': int(prediction)})

        except Exception as e:
            # Return error message as JSON
            return jsonify({'error': str(e)}), 500

    # Render the HTML page for GET requests
    return render_template('page6.html')

@app.route('/page11', methods=['GET', 'POST'])
def page11():
    if request.method == 'POST':
        try:
            # Extract JSON data from the request
            data = request.json

            # Extract features 
            features = [
                data.get('age', 0),
                data.get('sex' , 0),
                data.get('cp', 0),
                data.get('trestbps' , 0),
                data.get('chol', 0),
                data.get('fbs', 0),
                data.get('restecg', 0),
                data.get('thalach' , 0),
                data.get('exang' , 0),
                data.get('oldpeak' , 0),
                data.get('slope' , 0),
                data.get('ca' , 0),
                data.get('thal', 0)

            ]

    
            # Convert features to numpy array and reshape for the model
            features_array = np.array([features])
            print(features_array)

            # Make prediction
            prediction = Hypertension_model.predict(features_array)[0]

            # Return prediction as JSON
            return jsonify({'Hypertension': int(prediction)})

        except Exception as e:
            # Return error message as JSON
            return jsonify({'error': str(e)}), 500

    # Render the HTML page for GET requests
    return render_template('page11.html')


@app.route('/page12', methods=['GET', 'POST'])
def page12():
    if request.method == 'POST':
        try:
            # Extract JSON data from the request
            data = request.json

            # Extract features and convert to integers
            features = [
               data.get('Age', 0),
               data.get('Gender', 0),
               data.get('Ethnicity', 0),
               data.get('Smoking', 0),
               data.get('PhysicalActivity', 0),
               data.get('DietQuality', 0),
               data.get('SleepQuality', 0),
               data.get('PollenExposure', 0),
               data.get('DustExposure', 0),
               data.get('FamilyHistoryAsthma', 0),
               data.get('HistoryOfAllergies', 0),
               data.get('Eczema', 0),
               data.get('HayFever', 0),
               data.get('GastroesophagealReflux', 0),
               data.get('LungFunctionFEV1', 0),
               data.get('Wheezing', 0),
               data.get('ChestTightness', 0),
               data.get('Coughing', 0),
               data.get('NighttimeSymptoms', 0),
               data.get('ExerciseInduced', 0),
               
        

            ]

            # Convert features to numpy array and reshape for the model
            features_array = np.array([features])
            print(features_array)

            # Make prediction
            prediction = Asthma_model.predict(features_array)[0]

            # Return prediction as JSON
            return jsonify({'Asthma': int(prediction)})

        except Exception as e:
            # Return error message as JSON
            return jsonify({'error': str(e)}), 500

    # Render the HTML page for GET requests
    return render_template('page12.html')


if __name__ == '__main__':
    app.run(debug=True)
