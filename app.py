import re
import base64
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from scipy.misc import imread, imresize
from keras.models import load_model
from joblib import load

app = Flask(__name__)

knn_model_restored = load('./models/knn_model.joblib')
svm_model_restored = load('./models/svm_model.joblib')
mlp_model = load_model("./models/mlp_model.h5")
conv2d_model = load_model("./models/conv2d_model.h5")
graph = tf.get_default_graph()


# decoding an image from base64 into raw representation
def convert_image(img_data):
    imgstr = re.search(r'base64,(.*)', str(img_data)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


def make_prediction(image, model_name):

    assert model_name in ('knn', 'svm', 'mlp', 'conv2d')

    if model_name == 'knn':
        y_pred = knn_model_restored.predict(image)[0]
    elif model_name == 'svm':
        y_pred = svm_model_restored.predict(image)[0]
    elif model_name == 'mlp':
        with graph.as_default():
            y_pred = mlp_model.predict_classes(image)[0]
    elif model_name == 'conv2d':
        with graph.as_default():
            y_pred = conv2d_model.predict_classes(image.reshape(-1,28,28,1))[0]
    return y_pred


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    img_data = request.get_data()

    # encode it into a suitable format
    convert_image(img_data)

    # read the image into memory
    x = imread('output.png', mode='L')
    # make it the right size
    x = imresize(x, (28, 28))

    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 784)
    x = x.astype('float32')
    x = x / 255.0

    y_pred = make_prediction(x, 'conv2d')
    print(y_pred)
    return str(y_pred)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
