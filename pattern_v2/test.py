import torch
import torchvision
import torchvision.transforms as T
import cv2
import os
import numpy as np
from PIL import Image

# 저장된 모델 불러오기
MODEL_PATH = "ssd_model_optimized.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SSD 모델 로드
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
num_classes = 3  # 배경(0) + 체크무늬(1) + 단색(2)
model.head.classification_head.num_classes = num_classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # 평가 모드로 설정

# 데이터 변환 정의
transform = T.Compose([
    T.Resize((300, 300)),  # 🔥 이미지 크기를 모델 입력 크기(300x300)로 변환
    T.ToTensor(),
])

# 테스트 이미지 폴더 및 결과 저장 폴더
TEST_FOLDER = "test"
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 하나의 결과 파일 생성
RESULT_TXT_PATH = os.path.join(RESULT_FOLDER, "predictions.txt")

# 🔥 전체 결과 파일을 생성 및 덮어쓰기 모드로 열기
with open(RESULT_TXT_PATH, "w") as result_file:
    result_file.write("== SSD Detection Results ==\n\n")

    # 테스트 이미지 로드 및 예측
    for filename in os.listdir(TEST_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(TEST_FOLDER, filename)

            # 원본 이미지 로드 & 크기 변환
            image = Image.open(image_path).convert("RGB")
            image_resized = image.resize((300, 300))  # 🔥 OpenCV에서 좌표 매칭을 위해 300x300으로 변환
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # 예측 실행
            with torch.no_grad():
                predictions = model(image_tensor)

            # 🔍 예측된 정보 디버깅 출력
            print(f"🔹 Processing {filename}")

            # 원본 이미지 로드 (OpenCV) 및 크기 변환
            image_cv2 = cv2.imread(image_path)
            image_cv2 = cv2.resize(image_cv2, (300, 300))  # 🔥 크기 300x300으로 변환

            # 바운딩 박스 그리기 및 결과 저장
            detected = False  # 바운딩 박스가 그려졌는지 체크
            result_file.write(f"Filename: {filename}\n")
            result_file.write("Bounding Boxes (x_min, y_min, x_max, y_max), Score, Label\n")

            for box, score, label in zip(predictions[0]['boxes'].cpu().numpy(),
                                         predictions[0]['scores'].cpu().numpy(),
                                         predictions[0]['labels'].cpu().numpy()):

                if score > 0.7:  # 신뢰도
                    detected = True
                    x_min, y_min, x_max, y_max = map(int, box)

                    # 🔥 체크무늬와 단색을 구분하여 색상 다르게 적용
                    if label == 1:  # Checkered
                        color = (0, 0, 255)       # 빨간색
                        label_text = "Checkered"
                    elif label == 2:  # Solid
                        color = (0, 255, 0)       # 초록색
                        label_text = "Solid"
                    elif label == 3:  # Stripe
                        color = (255, 0, 0)       # 파란색
                        label_text = "Stripe"
                    elif label == 4:  # Dotted
                        color = (0, 255, 255)     # 노란색 (청록)
                        label_text = "Dotted"
                    elif label == 5:  # Floral
                        color = (255, 0, 255)     # 자홍색
                        label_text = "Floral"
                    elif label == 6:  # Animal
                        color = (255, 255, 0)     # 하늘색
                        label_text = "Animal"
                    elif label == 7:  # Paisley
                        color = (128, 0, 255)     # 보라색 계열
                        label_text = "Paisley"
                    elif label == 8:  # Printed
                        color = (255, 128, 0)     # 주황색
                        label_text = "Printed"
                    elif label == 9:  # Camouflage
                        color = (0, 128, 128)     # 다크 청록색
                        label_text = "Camouflage"
                    elif label == 10:  # Text
                        color = (128, 255, 0)     # 연두색
                        label_text = "Text"
                    elif label == 11:  # Logo
                        color = (0, 0, 128)       # 진한 빨강
                        label_text = "Logo"
                    elif label == 12:  # Pocket
                        color = (128, 128, 128)   # 회색
                        label_text = "Pocket"
                    else:
                        continue  # 배경(0) 무시


                    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), color, 1)
                    #cv2.putText(image_cv2, f"{label_text}: {score:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    text = f"{label_text}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image_cv2, (x_min, y_min - text_height - 4), (x_min + text_width, y_min), color, -1)
                    cv2.putText(image_cv2, text, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 🔥 TXT 파일에 바운딩 박스 정보 저장
                    result_file.write(f"{x_min}, {y_min}, {x_max}, {y_max}, {score:.2f}, {label_text}\n")

            # 바운딩 박스가 없을 경우 경고 출력
            if not detected:
                print(f"⚠️ No valid detections for {filename}")
                result_file.write("No valid detections\n")

            # 결과 이미지 저장
            result_path = os.path.join(RESULT_FOLDER, filename)
            cv2.imwrite(result_path, image_cv2)
            print(f"✅ Saved: {result_path}")

print(f"📄 모든 테스트 이미지 결과가 {RESULT_TXT_PATH} 에 저장되었습니다!")
