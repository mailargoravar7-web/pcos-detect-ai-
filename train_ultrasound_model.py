import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "ultrasound_dataset_ready")

TRAIN = os.path.join(DATASET, "train")
VAL = os.path.join(DATASET, "validation")
TEST = os.path.join(DATASET, "test")

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

# ======================================================
# DATA
# ======================================================

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    VAL,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = val_gen.flow_from_directory(
    TEST,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(train_data.class_indices)

# ======================================================
# MODEL
# ======================================================

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.30)(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.20)(x)

output = Dense(1,activation="sigmoid")(x)

model = Model(base_model.input,output)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
# CALLBACKS
# ======================================================

checkpoint = ModelCheckpoint(
    "ultrasound_model.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3
)

# ======================================================
# PHASE 1
# ======================================================

print("\nPHASE 1 TRAINING\n")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[checkpoint,early,reduce]
)

# ======================================================
# PHASE 2
# ======================================================

print("\nPHASE 2 FINE TUNING\n")

base_model.trainable=True

for layer in base_model.layers[:-30]:
    layer.trainable=False

model.compile(
    optimizer=Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[checkpoint,early,reduce]
)

# ======================================================
# TEST
# ======================================================

loss,acc=model.evaluate(test_data)

print("="*50)
print("TEST ACCURACY :",acc*100)
print("="*50)