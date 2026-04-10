from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load the trained Wine classification model
model = tf.keras.models.load_model('my_model.keras')
class_labels = ['Class 1', 'Class 2', 'Class 3']

@app.route('/')
def home():
    return "Welcome to the Wine Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            input_data = np.array([[
                float(data['alcohol']),
                float(data['malic_acid']),
                float(data['ash']),
                float(data['alcalinity_of_ash']),
                float(data['magnesium']),
                float(data['total_phenols']),
                float(data['flavanoids']),
                float(data['nonflavanoid_phenols']),
                float(data['proanthocyanins']),
                float(data['color_intensity']),
                float(data['hue']),
                float(data['od280_od315']),
                float(data['proline'])
            ]])

            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)