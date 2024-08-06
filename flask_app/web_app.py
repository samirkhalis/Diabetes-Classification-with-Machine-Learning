from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('logistic_regression.pkl')

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json(force=True)
        # Convert data into numpy array
        features = np.array(
            [data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'], data['Insulin'],
             data['BMI'], data['DiabetesPedigreeFunction'], data['Age']])

        # Make prediction using model
        prediction = model.predict([features])
        # Return prediction as JSON response
        output = prediction[0]

        return jsonify({'result': int(output)})

    except KeyError as e:
        return jsonify({'error': f'Missing or incorrect data key: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
