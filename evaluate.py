from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("pcos_model.keras")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "cnn_dataset/test",   # <-- Use your actual test folder
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")