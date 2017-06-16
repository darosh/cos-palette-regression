from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras import optimizers

import numpy as np
import math
import os
from encoder import Encoder

seed = 1
batch_size = 64
# epochs = 128*2
epochs = 16
width = 64
# samples = 1024*16
samples = 1024
split = 0.5
params = 4
freq = 4
clamping = False

show_margin = 16
show_height = 11
show_space = 1
show_sep = 12
show_text = 144 + 8

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(seed)

def fnc(p, t):
    return p[0] + p[1] * math.cos(2 * math.pi * (t * p[2] * freq + p[3]))

def clamp(my_value, max_value=1, min_value=0):
    return max(min(my_value, max_value), min_value)

def grade(value):
    return int(255 * clamp(value))

def get_samples(samples, xx=np.array([])):
    x = xx if xx.any() else np.random.random((samples, 4))
    y = np.zeros((samples, width, 2))
    i = 0
    while i < samples:
        j = 0;
        while j < width:
            t = j / width
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

def save(name='cpr'):
    model.save_weights(name+'.hdf5')
    with open(name+'.json', 'w') as f:
        f.write(model.to_json())
    encoder = Encoder(name+'.hdf5')
    encoder.serialize()
    encoder.save()

def show(x_train, y_train, x_pred, y_pred, samples = 10):
    from PIL import Image, ImageDraw

    w = show_margin * 2 + width + show_margin + show_text
    h = show_margin * 2 + samples * (2 * show_height + show_space) + (samples - 1) * show_sep

    img = Image.new( 'RGB', (w,h), 'white')
    draw = ImageDraw.Draw(img)
    pixels = img.load()

    i = 0;
    while i < samples:
        draw.text((show_margin * 2 + width, show_margin + i * (2 * show_height + show_space) + i * show_sep), np.array2string(y_train[i], precision=2, suppress_small=True), fill=(0,0,0,255))
        draw.text((show_margin * 2 + width, show_margin + i * (2 * show_height + show_space) + i * show_sep + show_height + show_space), np.array2string(y_pred[i], precision=2, suppress_small=True), fill=(0,0,0,255))

        j = 0;
        while j < show_height:
            y = show_margin + i * (2 * show_height + show_space) + i * show_sep + j
            for k in x_train[i]:
                x = show_margin + math.floor(width * k[0])
                c = grade(k[1])
                pixels[x,y] = (c, c, c)
            j += 1

        j = 0;
        while j < show_height:
            y = show_margin + i * (2 * show_height + show_space) + i * show_sep + j + show_height + show_space
            for k in x_pred[i]:
                x = show_margin + math.floor(width * k[0])
                c = grade(k[1])
                pixels[x,y] = (c, c, c)
            j += 1

        i += 1

    img.show()

(x_train, y_train) = get_samples(samples)
model = get_model()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=split)
save()
y_pred = get_predicted(model, samples, x_train)
(x_pred, y_pred) = get_samples(samples, y_pred)

show(x_train, y_train, x_pred, y_pred, samples = 20)
