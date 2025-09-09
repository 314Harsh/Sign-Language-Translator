import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import json

# -------------------
# CONFIG
# -------------------
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"
MODEL_PATH = "model/sign_language_model.h5"
LABELS_PATH = "model/labels.json"

# -------------------
# DATASET
# -------------------
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# -------------------
# MODEL
# -------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -------------------
# TRAIN
# -------------------
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=[checkpoint])

# -------------------
# SAVE LABELS
# -------------------
labels = {v: k for k, v in train_gen.class_indices.items()}
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f)

print("✅ Training complete. Model saved to:", MODEL_PATH)
