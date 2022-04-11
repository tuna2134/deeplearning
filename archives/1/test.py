from keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

img = load_img("image.jpg", target_size=(28, 28))
x = img_to_array(img)

model = keras.models.load_model('model.h5')
print(model.evaluate(x_test, y_test, verbose=2))
model.predict(np.array([x]))