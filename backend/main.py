from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
# translator = pipeline("translation", model="facebook/m2m100_418M", src_lang="en", tgt_lang="ko")

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    result = image_to_text(img)
    # kresult = translator(result)
    return result
    # return kresult