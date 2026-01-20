from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model with the new filename
model_filename = 'breast_cancer_model.pkl'

if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
else:
    # Updated error message to reflect the new script name
    print(f"Error: '{model_filename}' not found. Please run 'python model_building.py' first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model not loaded. Please contact administrator.')

    try:
        # Get data from form and convert to float
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # 0 = Malignant, 1 = Benign
        output = 'Malignant (Cancerous)' if prediction[0] == 0 else 'Benign (Safe)'
        
        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    # Render assigns a port via the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)