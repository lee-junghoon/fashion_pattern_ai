import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from PIL import Image
from torchvision.ops import sigmoid_focal_loss

# 저장할 JSON 파일 및 이미지 폴더
ANNOTATION_FILE = "annotations.json"
IMAGE_FOLDER = "dataset"
TARGET_SIZE = (300, 300)  # 모델 입력 크기
BATCH_SIZE = 4  # 배치 크기 증가
NUM_EPOCHS = 50  # 학습 epoch 증가
LEARNING_RATE = 0.0003  # 학습률 감소
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 변환 정의 (Data Augmentation 추가)
transform = T.Compose([
    T.Resize((300, 300)),  # 이미지 크기 조정
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 색상 변형
    T.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    T.RandomRotation(degrees=15),  # 15도 이내 랜덤 회rorcpfmf xkawl gkrl dnlgo전
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 위치 이동
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# COCO 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, annotation_file, image_folder, transform=None):
        with open(annotation_file, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image_info = self.data["images"][idx]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_folder, file_name)
        image = Image.open(image_path).convert("RGB")

        # 원본 이미지 크기
        orig_width, orig_height = image.size

        # 바운딩 박스 데이터 가져오기
        valid_boxes = []
        valid_labels = []
        for ann in self.data["annotations"]:
            if ann["image_id"] == image_info["id"]:
                x, y, w, h = ann["bbox"]

                category_id = ann["category_id"]
                # # 바운딩 박스 크기 변환 비율 계산
                # scale_x = TARGET_SIZE[0] / orig_width
                # scale_y = TARGET_SIZE[1] / orig_height

                # # 바운딩 박스 좌표 변환
                # new_x = x * scale_x
                # new_y = y * scale_y
                # new_w = w * scale_x
                # new_h = h * scale_y

                # ✅ 너비와 높이가 0보다 큰 경우만 추가
                if w > 0 and h > 0:
                    valid_boxes.append([x, y, x + w, y + h])
                    valid_labels.append(category_id)  

        # ✅ 바운딩 박스가 없는 경우 해당 이미지 학습에서 제외
        if len(valid_boxes) == 0:
            return None  # 🚨 바운딩 박스가 없는 이미지는 None 반환

        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # 이미지 변환 적용
        if self.transform:
            #image = F.resize(image, TARGET_SIZE)  # 이미지 크기 조정 후 반환
            image = F.to_tensor(image)  # 텐서 변환

        return image, target

def collate_fn(batch):
    batch = [data for data in batch if data is not None]  # 🚨 None 제거
    if len(batch) < 2:
        return [], []  # ✅ 빈 배치 방지

    images, targets = zip(*batch)  # ✅ (image, target) 형태로 Unpack
    return list(images), list(targets)

# 데이터셋 및 데이터로더 준비
train_dataset = CustomDataset(ANNOTATION_FILE, IMAGE_FOLDER, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# SSD 모델 불러오기 (모델 변경: 작은 객체 탐지 개선)
import torchvision
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True) 
num_classes = 3  # 배경(0) + 체크 패턴(1)  + 단색 패턴(2)
model.head.classification_head.num_classes = num_classes
model.to(DEVICE)

# 옵티마이저 및 학습률 스케줄러 추가
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 10 epoch마다 학습률 감소

# 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        if len(images) == 0:
            continue

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

# 학습된 모델 저장
MODEL_PATH = "ssd_model_optimized.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
