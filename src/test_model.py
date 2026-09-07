import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
MODEL_PATH = "../models/best_model.h5"
IMG_PATH = "sample.jpeg"

# --- Class Names (MUST match your training order) ---
class_names = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']

# --- Load Model ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load & Preprocess Image ---
img = image.load_img(IMG_PATH, target_size=(224, 224))
arr = image.img_to_array(img) / 255.0
arr = np.expand_dims(arr, axis=0)

# --- Predict ---
pred = model.predict(arr)
class_idx = np.argmax(pred)
class_prob = np.max(pred)

# --- Print results in console ---
print(f"\nPredicted Disease: {class_names[class_idx]}")
print(f"Confidence: {class_prob:.2f}")

# --- Display the image with prediction ---
plt.figure(figsize=(5, 5))
plt.imshow(image.load_img(IMG_PATH))
plt.axis("off")
plt.title(f"{class_names[class_idx]} ({class_prob*100:.1f}% confidence)")
plt.show()
