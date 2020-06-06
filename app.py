
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from scipy.spatial import distance
from skimage.transform import resize
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
global graph
graph = tf.get_default_graph()
image_size = 160
cascade_path = 'haarcascade_frontalface_alt2.xml'
model_path = 'facenet_keras.h5'
model = load_model(model_path)






def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def face_align(img, margin=10):
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img,
                                     scaleFactor=1.1,
                                     minNeighbors=3)

    (x, y, w, h) = faces[0]
    cropped = img[y - margin // 2:y + h + margin // 2,
              x - margin // 2:x + w + margin // 2, :]
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    return aligned


def encoding(img, model, graph):
    with graph.as_default():
        pred = model.predict(np.expand_dims(img, axis=0))
        emb = l2_normalize(pred)
        return emb


def get_diff(img1, model, graph):
    # model = load_model(model_path)

    img2 = cv2.imread('shahin1.jpg', cv2.IMREAD_COLOR)

    image1 = prewhiten(face_align(img1))
    image2 = prewhiten(face_align(img2))

    embs1 = encoding(image1, model, graph)
    embs2 = encoding(image2, model, graph)
    # k.clear_session()
    return distance.euclidean(embs1, embs2)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():

    r = request
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
    return jsonify(get_diff(img, model, graph))



if __name__ == "__main__":
    app.run(debug=True)