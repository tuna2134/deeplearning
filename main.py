import json
import keras
import numpy as np
from os import listdir
from keras import layers
from keras.preprocessing.image import load_img, img_to_array

def build_model(shape):
    inputs = keras.Input(shape=shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def get_data():
    x = []
    y = []
    path = "images"
    for n, name in enumerate(listdir("images")):
        for i in listdir(path + "/" + name):
            img = load_img(path + "/" + name+ "/" + i, target_size=(128, 128))
            x.append(img_to_array(img))
            y.append(n)
    return np.array(x), np.array(y)

train, label = get_data()

model = build_model((128, 128, 3))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, label, epochs=10)
img = load_img("images/dogs/000001.jpg", target_size=(128, 128))
print(model.predict(np.array([img_to_array(img)]))[0].argmax())
model.save("model.h5")