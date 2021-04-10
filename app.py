import os
import numpy as np
from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import pickle
from keras.models import load_model

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

model = load_model('model/age_model.h5')
with open('model/labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)
with open('model/age_dict.pkl', 'rb') as handle:
    age_dict = pickle.load(handle)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            test_image = image.load_img(f'{PEOPLE_FOLDER}/{filename}', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)

            position_result = np.argmax(result)
            label_result = list(labels.keys())[position_result]
            prob_result = float(result[0][position_result])

            position_result = np.argmax(result)
            label_result = list(labels.keys())[position_result]
            prob_result = float(result[0][position_result])
            response_text = f'La persona posee {age_dict[label_result]} con un {(prob_result*100):.2f}% de probabilidad!'
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('index.html', prediction_text = response_text, user_image = full_filename)
    return render_template('index_base.html')

if __name__ == "__main__":
    app.run(debug = True)
