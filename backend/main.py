from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model import predict_a_image
import io



app=FastAPI(title="이미지 텍스트 변환")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    label_idx, label_name, confidence = predict_a_image(image)

    return {
        "label_name" : label_name,
        "confidence": confidence
    }

# uvicorn main:app --port=8081 --reload