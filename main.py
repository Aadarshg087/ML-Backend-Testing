from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_leaf_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes):
    # Open the image
    image = Image.open(io.BytesIO(image_bytes))

    # Resize image
    image = image.resize((224, 224))  # Adjust based on model requirements

    # Convert to RGB if image is not already in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize
    image = np.array(image, dtype=np.float32) / 255.0

    # Print input details for debugging
    print("Input details shape:", input_details[0]['shape'])
    print("Image shape before inference:", image.shape)

    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Preprocess image
    image = preprocess_image(image_bytes)

    # Add batch dimension if not already present
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Ensure image matches input tensor shape
    input_shape = input_details[0]['shape']

    # Correctly compare shapes
    if not np.array_equal(image.shape, input_shape):
        print(f"Reshaping image from {image.shape} to {input_shape}")
        # Reshape to match exact input shape
        image = image.reshape(input_shape)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)

    return {"prediction": int(predicted_class)}

# Optional: Add error handling and logging
@app.on_event("startup")
async def startup_event():
    print("Input details:", input_details)
    print("Output details:", output_details)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}