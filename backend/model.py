import torch.nn as nn
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import json


LABEL_TYPE='coarse' # 세분류 말고 대분류 범주 사용
IMG_SIZE=128
BATCH_SIZE=16
EPOCHS=30
NUM_CLASSES=43 if LABEL_TYPE == 'coarse' else 81 # coarse 라벨 기준


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "grocery_cnn_coarse.pth" # 학습 끝난 모델 파라미터 위치
CLASS_PATH = BASE_DIR / "models" / "class_names_coarse.json" # 각 인덱스/라벨 이름 매핑 json


# 인덱스로 접근하여 한글/영문 클래스 이름 추출
with open(CLASS_PATH, "r", encoding="utf-8") as f:
   CLASS_NAMES = json.load(f)

device=torch.device("cpu")



# cnn 모델 정의
class GroceryStoreCNN(nn.Module):
  def __init__(self, num_classes=NUM_CLASSES):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64,128, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128,256, 3, padding=1)
    self.bn4 = nn.BatchNorm2d(256)

    self.pool  = nn.MaxPool2d(2,2)
    self.relu  = nn.ReLU(inplace=True)

    self.gap   = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.4)

    self.fc1 = nn.Linear(256, 512)
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self,x):
      x = self.pool(self.relu(self.bn1(self.conv1(x))))
      x = self.pool(self.relu(self.bn2(self.conv2(x))))
      x = self.pool(self.relu(self.bn3(self.conv3(x))))
      x = self.pool(self.relu(self.bn4(self.conv4(x))))

      x = self.gap(x)
      x = x.view(x.size(0), -1)

      x = self.dropout(self.relu(self.fc1(x)))
      x = self.fc2(x)
      return x


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# 모델 로드
def load_model():
    model = GroceryStoreCNN(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

MODEL = load_model()


# 예측 함수
def predict_a_image(image: Image.Image): # 입력 : PIL.Image
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = MODEL(img)
        confidence = F.softmax(outputs, dim=1)
        top_con, top_idx = torch.max(confidence, dim=1)

    pred_idx = int(top_idx.item()) # 예측 클래스 인덱스
    confi = float(top_con.item()) # 신뢰도
    class_name = CLASS_NAMES[pred_idx] # 인덱스에 해당하는 클래스 이름

    return pred_idx, class_name, confi


