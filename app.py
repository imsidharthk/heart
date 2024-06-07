from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform([np.array(data)])
    prediction = model.predict(final_input)
    output = prediction[0]
    return render_template('index.html', prediction_text='Heart Disease Prediction: {}'.format('Positive' if output == 1 else 'Negative'))

if __name__ == "__main__":
    app.run(debug=True)
