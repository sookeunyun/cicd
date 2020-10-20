from flask import Flask, send_file

import matplotlib.pyplot as plt
import io
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
from smart_open import open

app = Flask(__name__)

#################################
# 모델을 S3로부터 불러옵니다.
#################################
MODEL = None
def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    activate = 'softmax'
    dropout = float(0)

    batch_size, num_classes, hidden = (128, 10, 512)
    loss_func = "categorical_crossentropy"
    # smart_open을 이용하여 S3에서 모델파일을 불러옵니다.
    with open('s3://mlops-2020/ysk1438/mymodel.npy', 'rb') as f:
        weights = np.load(f, allow_pickle=True)
        # build model
        model = Sequential()
        model.add(Dense(hidden, activation='relu', input_shape=(784,)))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation=activate))

        model.compile(loss=loss_func, optimizer=RMSprop(), metrics=['accuracy'])
        model.set_weights(weights)
    MODEL = model
    return model


##########################
# 숫자를 예측합니다.
##########################
@app.route('/predict/<num>')
def predict(num):
    # 숫자에 대응되는 numpy 이미지를 불러옵니다.
    x = get_image(int(num))

    # 모델 예측하기
    with tf.get_default_graph().as_default():
        model = get_model()
        y = model.predict(x.reshape(1, 784))
    print("Predict: ", np.argmax(y))

    mem = io.BytesIO()
    plt.imsave(mem, x)
    mem.seek(0)

    return send_file(mem, mimetype="image/png")


# 숫자에 대응하는 이미지를 불러옵니다.
def get_image(num):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    idx = [3, 2,1,18,4,15,11,0,61,7]
    return x_test[idx[num]]


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
