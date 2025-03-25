import argparse
import subprocess
import sys
import os
import signal
import time
import logging
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import io
import requests
import uvicorn
import json

# 설정
MODEL_PATH = "../ssd_model_optimized.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PID_FILE = "inference.pid"
LOG_DIR = "logs"

# 로그 설정
def setup_logger():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "inference.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# 모델 로딩
def load_model():
    logger.info("모델 로딩 중...")
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model.head.classification_head.num_classes = 13
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info("모델 로딩 완료.")
    return model

model = load_model()

# 라벨 매핑
LABEL_MAP = {
    1: "Checkered", 2: "Solid", 3: "Stripe", 4: "Dotted",
    5: "Floral", 6: "Animal", 7: "Paisley", 8: "Printed",
    9: "Camouflage", 10: "Text", 11: "Logo", 12: "Pocket"
}

# 전처리
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
])

# FastAPI 앱
app = FastAPI()

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
        logger.error(f"이미지 로드 실패: {e}")
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

    log_data = {
        "image_url": request.image_url,
        "predictions": result
    }

    logger.info("예측 결과: %s", json.dumps(log_data, ensure_ascii=False))
    return log_data

# 백그라운드 실행 함수
def start_server():
    if os.path.exists(PID_FILE):
        print("이미 서버가 실행 중입니다.")
        return
    command = [sys.executable, __file__, "--run"]
    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))
    print(f"서버 시작됨. PID: {process.pid}")

def stop_server():
    if not os.path.exists(PID_FILE):
        print("실행 중인 서버가 없습니다.")
        return
    with open(PID_FILE, "r") as f:
        pid = int(f.read())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"서버 중지됨. PID: {pid}")
    except Exception as e:
        print(f"서버 종료 실패: {e}")
    finally:
        os.remove(PID_FILE)

# 직접 실행용
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true", help="서버 시작")
    parser.add_argument("--stop", action="store_true", help="서버 중지")
    parser.add_argument("--run", action="store_true", help="내부 실행")
    args = parser.parse_args()

    if args.start:
        start_server()
    elif args.stop:
        stop_server()
    elif args.run:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
