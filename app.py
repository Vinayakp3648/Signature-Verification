from flask import Flask,request,render_template,redirect,url_for
import tensorflow as tf
import PIL
import numpy as np
import os
from utils import preprocess_image
import pickle
from utils import plot_history

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Define the upload folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'C:/Users/HP/Desktop/MINI PROJECT 6TH SEM/To_sent/To_sent/saved_model/my_model.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/',methods=['GET','POST'])

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            image = preprocess_image(file_path)

            print(f"File path: {file_path}")
            print(f"Uploaded file name: {file.filename}")
            print(f"Preprocessed image shape: {image.shape}")

            # Predict using the model
            prediction = model.predict(image)
            # Print predictions for debugging
            print(f"Predictions: {prediction}")



            prediction = model.predict(image)

            real_forged_prob = prediction[0][0][0]
            individual_prob = prediction[1][0]
            predicted_individual = np.argmax(individual_prob)

            labels = [f'Signature {i}' for i in range(1,13)]
            predicted_label = labels[predicted_individual]

            is_real = 'REAL' if real_forged_prob > 0.5 else 'FORGED'

            return render_template('result.html',label = predicted_label, is_real = is_real)
    return render_template('upload.html')

@app.route('/plot')
def plot():
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)
    plot_history(history)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

