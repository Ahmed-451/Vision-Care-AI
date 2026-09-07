import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "../models/best_model.h5"
IMG_PATH = "../data/test/sample.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(IMG_PATH, target_size=(224, 224))
arr = image.img_to_array(img) / 255.0
arr = np.expand_dims(arr, 0)

pred = model.predict(arr)
print("Predicted class index:", np.argmax(pred))
