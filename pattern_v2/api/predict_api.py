from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import io
import requests

# 모델 로딩
MODEL_PATH = "../ssd_model_optimized.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.head.classification_head.num_classes = 13  # 0 = background + 12 classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 라벨 매핑
LABEL_MAP = {
    1: "Checkered",
    2: "Solid",
    3: "Stripe",
    4: "Dotted",
    5: "Floral",
    6: "Animal",
    7: "Paisley",
    8: "Printed",
    9: "Camouflage",
    10: "Text",
    11: "Logo",
    12: "Pocket"
}

# 이미지 전처리
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
])

# FastAPI 앱 생성
app = FastAPI()

# 요청 모델
class PredictRequest(BaseModel):
    image_url: str
    score_threshold: float = 0.7

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(image_tensor)

    result = []
    for box, score, label in zip(predictions[0]["boxes"].cpu().numpy(),
                                  predictions[0]["scores"].cpu().numpy(),
                                  predictions[0]["labels"].cpu().numpy()):
        if score >= request.score_threshold and label in LABEL_MAP:
            result.append({
                "label": LABEL_MAP[label],
                "score": round(float(score), 4),
                "box": [round(float(x), 2) for x in box]
            })

    return {
        "image_url": request.image_url,
        "predictions": result
    }
