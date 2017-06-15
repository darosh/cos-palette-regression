from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras import optimizers

import numpy as np
import math
import os
from encoder import Encoder

seed = 1
batch_size = 64
epochs = 128
# epochs = 1
width = 64
samples = 1024*8
# samples = 200
split = 0.5
params = 4
freq = 32
clamping = False

show_margin = 8
show_space = 4
show_sep = 4

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(seed)

def fnc(p, t):
    return p[0] + p[1] * math.cos(freq * 2*math.pi * t * p[2] + p[3])

def clamp(my_value, max_value=0, min_value=1):
    return max(min(my_value, max_value), min_value)

def get_samples(samples, xx=np.array([])):
    x = xx if xx.any() else np.random.random((samples, 4))
    y = np.zeros((samples, width, 2))
    i = 0
    while i < samples:
        j = 0;
        while j < width:
            t = j / (width - 1)
            y[i][j][0] = t
            y[i][j][1] = clamp(fnc(x[i], t)) if clamping else fnc(x[i], t)
            j += 1
        i += 1
    return (y, x)

def get_predicted(model, samples, x):
    return model.predict(x)

def get_model():
    model = Sequential([
        Dense(width, activation='tanh', input_shape=(width, 2)),
        Dense(width//2, activation='tanh'),
        Flatten(),
        Dense(width//4, activation='tanh'),
        Dense(4, activation='linear'),
        ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def save():
    model.save_weights('cos-palette.hdf5')
    with open('cos-palette.json', 'w') as f:
        f.write(model.to_json())
    encoder = Encoder('cos-palette.hdf5')
    encoder.serialize()
    encoder.save()

def show():
    from PIL import Image
    img = Image.new( 'RGB', (255,255), 'white')
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i,j] = (i, j, 100)
    img.show()

(x_train, y_train) = get_samples(samples)
model = get_model()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=split)
# save()
# show()
y_pred = get_predicted(model, samples, x_train)
(x_pred, y_pred) = get_samples(samples, y_pred)
