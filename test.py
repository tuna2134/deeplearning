from keras.datasets import mnist
import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.load_model('model.h5')
print(model.evaluate(x_test, y_test, verbose=2))