from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_leaf_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels (modify based on your dataset)
class_labels = {0: "Arjun Leaf", 1: "Tulsi", 2: "Neem", 3: "Basil", 4: "Curry Leaf"}

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
    if not np.array_equal(image.shape, input_shape):
        image = image.reshape(input_shape)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output)

    # Get the class name from the dictionary
    predicted_class_name = class_labels.get(predicted_index, "Unknown")


    return {"prediction": predicted_class_name}
@app.get("api/health")
async def health_check():
    return {"status": "healthy"}
