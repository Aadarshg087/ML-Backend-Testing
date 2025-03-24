from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()

# Load your pretrained model
model = torch.load("model.pkl", map_location=torch.device("cpu"))
model.eval()    

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Adjust according to your model
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File()):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()

    return {"prediction": predicted_class}
