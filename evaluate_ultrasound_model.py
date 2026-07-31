import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "ultrasound_model.keras"
TEST_PATH = "ultrasound_dataset_ready/test"

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
