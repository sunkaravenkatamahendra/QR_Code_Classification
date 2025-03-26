# Flask Web App for QR Code Classification
from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("qr_classifier.h5")

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    return "Original" if prediction[0][0] < 0.5 else "Counterfeit"



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            # Ensure the 'static' directory exists
            if not os.path.exists("static"):
                os.makedirs("static")

            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            result = process_image(file_path)
            return render_template("index.html", result=result, image=file.filename)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
