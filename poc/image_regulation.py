import os
from PIL import Image

# 데이터셋 폴더 및 저장 폴더 경로
DATASET_FOLDER = "dataset"
TARGET_SIZE = (300, 300)  # 이미지 정규화 크기

# 저장할 폴더 (덮어쓰지 않고 원본 유지하고 싶다면 새 폴더 사용)
OUTPUT_FOLDER = "dataset_resized"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 이미지 변환 실행
for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # ✅ JPG, PNG 파일만 처리
        image_path = os.path.join(DATASET_FOLDER, filename)
        image = Image.open(image_path).convert("RGB")  # ✅ RGB로 변환
        image_resized = image.resize(TARGET_SIZE, Image.LANCZOS)  # ✅ 300x300 리사이즈 (고품질)

        # 변환된 이미지 저장
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        image_resized.save(output_path)
        print(f"✅ Resized and saved: {output_path}")

print("🎯 모든 이미지가 300x300으로 정규화되었습니다!")
