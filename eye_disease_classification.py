# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================

# Define paths (adjust these according to your dataset location)
data_dir = '/path/to/your/eye_disease_dataset'  # Change this path
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Get class names
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"\nNumber of Classes: {num_classes}")
print(f"Class Names: {class_names}")


# Count images per class
def count_images(directory):
    counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len(os.listdir(class_path))
    return counts


train_counts = count_images(train_dir)
print("\nTraining Images per Class:")
for class_name, count in train_counts.items():
    print(f"  {class_name}: {count}")

# ============================================
# STEP 2: DATA PREPROCESSING AND AUGMENTATION
# ============================================

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Only rescaling for validation and test sets
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")


# ============================================
# STEP 3: VISUALIZE SAMPLE IMAGES
# ============================================

def plot_sample_images(generator, class_names, num_samples=9):
    """Plot sample images from the dataset"""
    plt.figure(figsize=(15, 15))

    # Get a batch of images
    images, labels = next(generator)

    for i in range(min(num_samples, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        plt.title(f"Class: {class_names[class_idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_sample_images(train_generator, class_names)


# ============================================
# STEP 4: BUILD THE MOBILENETV2 MODEL
# ============================================

def create_mobilenet_model(num_classes, img_height=224, img_width=224):
    """
    Create a MobileNetV2 model with transfer learning
    """
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Create the full model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model, base_model


# Create the model
model, base_model = create_mobilenet_model(num_classes, IMG_HEIGHT, IMG_WIDTH)

# Display model architecture
model.summary()

print(f"\nTotal layers in base model: {len(base_model.layers)}")
print(f"Trainable layers: {len([layer for layer in base_model.layers if layer.trainable])}")

# ============================================
# STEP 5: COMPILE THE MODEL
# ============================================

# Compile the model
initial_learning_rate = 0.001

model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\nModel compiled successfully!")

# ============================================
# STEP 6: SET UP CALLBACKS
# ============================================

# Create callbacks
checkpoint = ModelCheckpoint(
    'best_eye_disease_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ============================================
# STEP 7: TRAIN THE MODEL (PHASE 1 - FROZEN BASE)
# ============================================

print("\n" + "=" * 50)
print("PHASE 1: Training with frozen base model")
print("=" * 50 + "\n")

EPOCHS_PHASE1 = 20

history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 8: FINE-TUNING (PHASE 2 - UNFREEZE LAYERS)
# ============================================

print("\n" + "=" * 50)
print("PHASE 2: Fine-tuning - Unfreezing top layers")
print("=" * 50 + "\n")

# Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Trainable layers after unfreezing: {len([layer for layer in base_model.layers if layer.trainable])}")

# Recompile with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

EPOCHS_PHASE2 = 20

history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks,
    verbose=1
)


# ============================================
# STEP 9: PLOT TRAINING HISTORY
# ============================================

def plot_training_history(history_phase1, history_phase2):
    """Plot training and validation metrics"""
    # Combine histories
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    plt.axvline(x=len(history_phase1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_training_history(history_phase1, history_phase2)

# ============================================
# STEP 10: EVALUATE ON TEST SET
# ============================================

print("\n" + "=" * 50)
print("EVALUATING ON TEST SET")
print("=" * 50 + "\n")

# Load the best model
model = keras.models.load_model('best_eye_disease_model.h5')

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Calculate F1 Score
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f"Test F1-Score: {f1_score * 100:.2f}%")

# ============================================
# STEP 11: PREDICTIONS AND CONFUSION MATRIX
# ============================================

# Get predictions
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50 + "\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Eye Disease Classification', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================
# STEP 12: PREDICTION FUNCTION FOR NEW IMAGES
# ============================================

def predict_eye_disease(image_path, model, class_names, img_size=(224, 224)):
    """
    Predict eye disease from a new image
    """
    # Load and preprocess image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # Display results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Input Image", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.barh(class_names, predictions[0] * 100)
    plt.xlabel('Confidence (%)', fontsize=12)
    plt.title('Prediction Probabilities', fontsize=14)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    return predicted_class, confidence


# Example usage:
# predicted_class, confidence = predict_eye_disease('path/to/test/image.jpg', model, class_names)


# ============================================
# STEP 13: SAVE THE MODEL
# ============================================

# Save the final model
model.save('final_eye_disease_mobilenetv2_model.h5')
print("\n✓ Final model saved as 'final_eye_disease_mobilenetv2_model.h5'")

# Save model in TensorFlow SavedModel format
model.save('saved_model/eye_disease_model')
print("✓ Model saved in TensorFlow SavedModel format")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)