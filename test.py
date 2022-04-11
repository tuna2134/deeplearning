from keras.preprocessing.image import load_img, img_to_array
import keras
import numpy as np

model = keras.models.load_model("model.h5")
img = load_img("images/cats/000003.jpg", target_size=(128, 128))
print(model.predict(np.array([img_to_array(img)]))[0].argmax())