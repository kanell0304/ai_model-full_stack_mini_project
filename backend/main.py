from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image

app=FastAPI(title="이미지 텍스트 변환")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

class GetImage(BaseModel):
    path:str

@app.post('/predict')
def predict(get_image:GetImage):
    img = Image.open(get_image.path)
    result = image_to_text(img)

    return(result)

# uvicorn main:app --port=8081 --reload