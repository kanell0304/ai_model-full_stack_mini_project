from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io
from typing import List, Dict
from contextlib import asynccontextmanager

# 모델 클래스 정의 (Jupyter Notebook과 동일)
class FruitVegClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)  # pretrained 대신 weights 사용
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = None
class_names = None
device = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, device, transform

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 클래스 이름 로드
        with open('class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        
        # 전처리 정보 로드
        with open('preprocessing.json', 'r') as f:
            prep_info = json.load(f)
        
        # 모델 구조 생성
        num_classes = len(class_names)
        model = FruitVegClassifier(num_classes)
        
        # state_dict 로드 (learned_fruit_veg_model.pth 사용)
        state_dict = torch.load('learned_fruit_veg_model.pth', map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        # Transform 정의
        transform = transforms.Compose([
            transforms.Resize((prep_info['image_size'], prep_info['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep_info['mean'], std=prep_info['std'])
        ])
        
        print(f"{device}로 작동하고 {len(class_names)}개의 클래스를 불러왔음")
        
    except Exception as e:
        print(f"모델을 불러오는데 오류가 발생했음: {e}")
        raise
    
    yield 

app = FastAPI(title="과일/채소 이미지 분류 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.get('/')
def root():
    return {"message": "과일/채소 분류 API", "status": "running"}

# 모델을 제대로 불러왔는지 체크
@app.get('/health')
def health_check():
    return {
        "status": "ready" if model is not None else "not ready",
        "device": str(device), # cpu/gpu 인지
        "num_classes": len(class_names) if class_names else 0, # 클래스 크기
        "classes": class_names # 클래스들 이름
    }


# 불러온 모델에 이미지를 업로드하여 예측 시키기
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None: # 불러온 모델이 없다면
        raise HTTPException(status_code=503, detail="Model not found")
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 전처리
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 가장 높은 점수의 예측 결과물 3개
            top3_prob, top3_idx = torch.topk(probabilities, min(3, len(class_names)))
            
            predictions = [
                {
                    "class": class_names[idx.item()], # 클래스이름
                    "confidence": round(prob.item() * 100, 2) # 정확도
                }
                for prob, idx in zip(top3_prob[0], top3_idx[0])
            ]
        
        return {
            "success": True, # 성공여부 (True/False)
            "predicted_class": predictions[0]["class"], # 예측한 클래스 이름 - Fruit/Vegetables
            "confidence": predictions[0]["confidence"], # 정확도
            "top_predictions": predictions # 상위 3개의 예측
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") # 예측에 실패했다면 오류 발생


# 지정된 클래스들 불러오기
@app.get('/classes')
def get_classes():
    if class_names is None:
        raise HTTPException(status_code=503, detail="Model not found")
    return {"classes": class_names}

# 실행: uvicorn main:app --port=8081 --reload
