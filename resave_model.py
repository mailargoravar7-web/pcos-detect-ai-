from tensorflow.keras.models import load_model

# Load your existing model
model = load_model("pcos_model.keras", compile=False)

# Save it again
model.save("pcos_model_render.keras")