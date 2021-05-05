import os
import numpy as np
from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import boto3

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

client = boto3.client('s3',
                    aws_access_key_id= os.environ['S3_KEY'],
                    aws_secret_access_key= os.environ['S3_SECRET'])

BUCKET_NAME='face-age-classification-storage'

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/photos'

model = tf.keras.models.load_model('model/age_model.h5', custom_objects={
                                       "KerasLayer": hub.KerasLayer})

with open('model/labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)

age_dict = {'1': 'entre 1 y 2 años',
            '2': 'entre 3 y 9 años',
            '3': 'entre 10 y 20 años',
            '4': 'entre 21 y 27 años',
            '5': 'entre 28 y 45 años',
            '6': 'entre 46 y 65 años',
            '7': 'sobre 65 años'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fullname = os.path.join(UPLOAD_FOLDER, filename)
            file.save(fullname)
            client.upload_file(Bucket = BUCKET_NAME, Filename = fullname, Key = filename)

            test_image = image.load_img(fullname, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)

            position_result = np.argmax(result)
            label_result = list(labels.keys())[position_result]
            prob_result = float(result[0][position_result])

            response_text = f'La persona posee {age_dict[label_result]} con un {(prob_result*100):.1f}% de probabilidad!'
            return render_template('index_base.html', prediction_text = response_text)
    return render_template('index_base.html')

if __name__ == "__main__":
    app.run(debug = True)
