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
async def predict(file: UploadFile = File(...)): # UploadFile로 받은 파일
   
    content = await file.read() # 데이터 읽음
    image = Image.open(io.BytesIO(content)).convert("RGB") # 메모리 상에서 파일처럼 감싸 Pillow 이미지 객체로 변환

    label_idx, label_name, confidence = predict_a_image(image) 
    # 이미지를 predict_a_image 함수에 전달


    # 모델이 예측한 label_index, label_name, confidence를 json으로 반환
    return {
        "label_name" : label_name,
        "confidence": confidence
    }



# uvicorn main:app --port=8081 --reload