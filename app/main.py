import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from app.Model_functions import get_diff
from keras.models import load_model

app = Flask(__name__)
app.config["SERVER_NAME"]="facenetapi.herokuapp.com"
model_path = 'app/modelFiles/facenet_keras.h5'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():

    r = request
    print(r.data)

    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # print(img.shape)
    #
    # response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
    #             }
    # # encode response using jsonpickle
    # response_pickled = jsonpickle.encode(response)

    # return Response(response=response_pickled, status=200, mimetype="application/json")
    return jsonify(get_diff(img , model))

