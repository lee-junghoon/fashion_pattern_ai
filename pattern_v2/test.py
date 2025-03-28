import torch
import torchvision
import torchvision.transforms as T
import cv2
import os
import json
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

# 누적 통계
total_files = 0
files_with_detections = 0
files_without_detections = 0
all_above_scores = []
all_below_scores = []

# annotations 불러오기
with open("annotation.json", "r") as f:
    annotations_data = json.load(f)

# 파일명 -> 실제 annotation 매핑용 dict 생성
gt_annotations = {}
category_id_to_name = {cat["id"]: cat["name"].capitalize() for cat in annotations_data["categories"]}

for ann in annotations_data["annotations"]:
    image_id = ann["image_id"]
    label_name = category_id_to_name[ann["category_id"]]
    if image_id not in gt_annotations:
        gt_annotations[image_id] = []
    gt_annotations[image_id].append(label_name)

# 🔥 전체 결과 파일을 생성 및 덮어쓰기 모드로 열기
with open(RESULT_TXT_PATH, "w") as result_file:
    result_file.write("== SSD Detection Results ==\n\n")
    
    #COCO format predictions 저장용 리스트
    coco_predictions = []

    # 테스트 이미지 로드 및 예측
    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            total_files += 1

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

            # 예측된 박스 정보 저장용
            above_scores = []
            below_scores = []

            for box, score, label in zip(predictions[0]['boxes'].cpu().numpy(),
                                         predictions[0]['scores'].cpu().numpy(),
                                         predictions[0]['labels'].cpu().numpy()):

                # 🔥 체크무늬와 단색을 구분하여 색상 다르게 적용
                if label == 1:  # Checked
                   color = (0, 0, 255)       # 빨간색
                   label_text = "Checked"
                elif label == 2:  # Solid
                   color = (0, 255, 127)       # 초록색
                   label_text = "Solid"
                elif label == 3:  # Stripe
                    color = (255, 0, 0)       # 파란색
                    label_text = "Stripe"
                elif label == 4:  # Dotted
                    color = (0, 255, 255)     # 노란색 (청록)
                    label_text = "Dotted"
                elif label == 5:  # Floral
                    color = (255, 105, 180)     # 핑크색
                    label_text = "Floral"
                elif label == 6:  # Animal
                    color = (173, 216, 230)   # 밝은 하늘색
                    label_text = "Animal"
                elif label == 7:  # Paisley
                   color = (186, 85, 211)    # 연보라
                   label_text = "Paisley"
                elif label == 8:  # Printed
                    color = (255, 165, 0)     # 오렌지
                    label_text = "Printed"
                elif label == 9:  # Camouflage
                    color = (60, 179, 113)    # 연녹색
                    label_text = "Camouflage"
                elif label == 10:  # Text
                    color = (255, 255, 0)     # 밝은 노랑
                    label_text = "Text"
                elif label == 11:  # Logo
                    color = (0, 0, 128)       # 진한 남색
                    label_text = "Logo"
                elif label == 12:  # Pocket
                    color = (192, 192, 192)   # 밝은 회색
                    label_text = "Pocket"
                else:
                    continue  # 배경(0) 무시
                
                if score > 0.7:  # 신뢰도
                    detected = True
                    above_scores.append(score)
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), color, 1)
                    #cv2.putText(image_cv2, f"{label_text}: {score:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    text = f"{label_text}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # 텍스트 배경 박스 (바운딩 박스 위쪽)
                    cv2.rectangle(image_cv2, (x_min, y_min - text_height - 8), (x_min + text_width, y_min), color, -1)
                    # 흰색 글자 출력
                    cv2.putText(image_cv2, text, (x_min, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # 🔥 TXT 파일에 바운딩 박스 정보 저장
                    result_file.write(f"{x_min}, {y_min}, {x_max}, {y_max}, {score:.2f}, {label_text}\n")

                    # COCO format용 box 추가
                    coco_predictions.append({
                        "image_id": filename,  # 또는 실제 image_id 값
                        "category_id": int(label),
                        "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                        "score": float(score)
                    })

                else: 
                    below_scores.append(score)

            all_above_scores.extend(above_scores)
            all_below_scores.extend(below_scores)

            # 바운딩 박스가 없을 경우 경고 출력
            if not detected:
                print(f"⚠️ No valid detections for {filename}")
                result_file.write("No valid detections\n")

            # 🔹 실제 라벨 정보 출력 (Ground Truth)
            image_ground_truths = gt_annotations.get(filename, [])
            result_file.write("Bounding box info:\n")
            if image_ground_truths:
                for idx, label_name in enumerate(image_ground_truths, start=1):
                    result_file.write(f" - GT Box {idx}: {label_name}\n")
            else:
                result_file.write(" - No ground truth annotations found\n")


            # 통계 정보 출력
            result_file.write("\nDetection Summary:\n")
            result_file.write(f" - Score >= 0.7: {len(above_scores)} boxes\n")
            result_file.write(f" - Score <  0.7: {len(below_scores)} boxes\n")
            if above_scores:
                result_file.write(f" - Avg score (>= 0.7): {np.mean(above_scores):.4f}\n")
            if below_scores:
                result_file.write(f" - Avg score (< 0.7): {np.mean(below_scores):.4f}\n")
            
            if len(above_scores) > 0:
                files_with_detections += 1
            else:
                files_without_detections += 1

            result_file.write("\n" + "="*40 + "\n\n")

            # 결과 이미지 저장
            result_path = os.path.join(RESULT_FOLDER, filename)
            cv2.imwrite(result_path, image_cv2)
            print(f"✅ Saved: {result_path}")
        else:
            print(f"⚠️ not Saved: {filename}")

    # 루프가 끝난 후 JSON 파일로 저장
    PRED_JSON_PATH = os.path.join(RESULT_FOLDER, "predictions_coco_format.json")
    with open(PRED_JSON_PATH, "w") as f:
        json.dump(coco_predictions, f, indent=4)
    print(f"📄 COCO format prediction saved to {PRED_JSON_PATH}")

    # 전체 통계 출력
    result_file.write("\nFinal Summary\n")
    result_file.write("=" * 40 + "\n")
    result_file.write(f"Total test files          : {total_files}\n")
    result_file.write(f"Files with score >= 0.7   : {files_with_detections}\n")
    result_file.write(f"Files with score <  0.7   : {files_without_detections}\n")
    if all_above_scores:
        result_file.write(f"Avg score (>= 0.7) overall: {np.mean(all_above_scores):.4f}\n")
    if all_below_scores:
        result_file.write(f"Avg score (<  0.7) overall: {np.mean(all_below_scores):.4f}\n")

print(f"📄 모든 테스트 이미지 결과가 {RESULT_TXT_PATH} 에 저장되었습니다!")
