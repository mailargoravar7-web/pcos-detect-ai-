# ==========================================================
# PCOS Detection using MobileNetV2 Transfer Learning
# TensorFlow 2.21.0
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = "cnn_dataset/train"
VALID_DIR = "cnn_dataset/validation"
TEST_DIR = "cnn_dataset/test"

# ==========================================================
# Hyperparameters
# ==========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 0.0001

# ==========================================================
# Data Augmentation
# ==========================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    zoom_range=0.20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    shear_range=0.20,

    horizontal_flip=True,

    fill_mode="nearest"

)

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

# ==========================================================
# Load Dataset
# ==========================================================

train_data = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary"

)

validation_data = validation_datagen.flow_from_directory(

    VALID_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary"

)

test_data = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

print("\n===============================")
print("DATASET INFORMATION")
print("===============================")

print("Training Images   :", train_data.samples)
print("Validation Images :", validation_data.samples)
print("Testing Images    :", test_data.samples)

print("\nClass Labels")

print(train_data.class_indices)

# ==========================================================
# Class Weights
# ==========================================================

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_data.classes),

    y=train_data.classes

)

class_weights = {

    0: class_weights[0],

    1: class_weights[1]

}

print("\nClass Weights")

print(class_weights)

# ==========================================================
# MobileNetV2 Base Model
# ==========================================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)

# Freeze pretrained layers

base_model.trainable = False

# ==========================================================
# Build Model
# ==========================================================

inputs = Input(shape=(224,224,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.40)(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(0.30)(x)

outputs = Dense(

    1,

    activation="sigmoid"

)(x)

model = Model(

    inputs,

    outputs

)

# ==========================================================
# Compile Model
# ==========================================================

optimizer = Adam(

    learning_rate=LEARNING_RATE

)

model.compile(

    optimizer=optimizer,

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

print("\n===============================")
print("MODEL SUMMARY")
print("===============================")

model.summary()
# ==========================================================
# Callbacks
# ==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# ==========================================================
# Training
# ==========================================================

print("\n===================================")
print("TRAINING STARTED")
print("===================================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    class_weight=class_weights,
    verbose=1
)

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================\n")

# ==========================================================
# Test Evaluation
# ==========================================================

print("\n===================================")
print("MODEL EVALUATION")
print("===================================\n")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# ==========================================================
# Predictions
# ==========================================================

print("\n===================================")
print("GENERATING PREDICTIONS")
print("===================================\n")

predictions = model.predict(
    test_data,
    verbose=1
)

predicted_classes = (
    predictions > 0.5
).astype("int32").flatten()

true_classes = test_data.classes

class_labels = list(
    test_data.class_indices.keys()
)
# ==========================================================
# Classification Report
# ==========================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels,
        zero_division=0
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================\n")

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    true_classes,
    predictions
)

roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC Score : {roc_auc:.4f}")

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")

plt.close()

# ==========================================================
# Accuracy Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("accuracy.png")

plt.close()

# ==========================================================
# Loss Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("loss.png")

plt.close()

# ==========================================================
# Save Final Model
# ==========================================================

model.save("pcos_model.keras")

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("Saved Files:")

print("✓ best_model.keras")
print("✓ pcos_model.keras")
print("✓ accuracy.png")
print("✓ loss.png")
print("✓ roc_curve.png")

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")

print(f"\nFinal Test Accuracy : {test_accuracy:.4f}")
print(f"Final Test Loss     : {test_loss:.4f}")
print(f"ROC-AUC Score       : {roc_auc:.4f}")