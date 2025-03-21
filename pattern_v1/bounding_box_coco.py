import cv2
import json
import os
import numpy as np

# 저장할 JSON 파일
ANNOTATION_FILE = "annotations.json"
# image_folder = "downloaded_images/checked"  # 이미지가 있는 폴더 경로
image_folder = "dataset"  # 이미지가 있는 폴더 경로

if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)
else:
    data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "checkered"}, {"id": 2, "name": "solid"}]}
image_id = max((img["id"] for img in data["images"]), default=0) + 1
annotation_id = max((ann["id"] for ann in data["annotations"]), default=0) + 1
bounding_boxes = []
current_image = None
backup_image = None
selected_category = 1 #기본값: checked

save_count = len(data['images'])  # 저장된 이미지 개수

drawing = False  # 드래그 상태 확인
ix, iy = -1, -1  # 드래그 시작 좌표
cursor_x, cursor_y = 0, 0  # 마우스 커서 위치

# 바운딩 박스 색상 리스트
BOX_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0)]

def draw_rectangle(event, x, y, flags, param):
    #global ix, iy, drawing, bounding_boxes, current_image, annotation_id, cursor_x, cursor_y
    global ix, iy, drawing, bounding_boxes, current_image, annotation_id, cursor_x, cursor_y, selected_category
    
    cursor_x, cursor_y = x, y  # 현재 마우스 위치 저장
    
    temp_image = current_image.copy()
    
    # 마우스 위치 가이드라인 추가 (회색 1픽셀)
    line_color = (192, 192, 192)
    cv2.line(temp_image, (cursor_x, 0), (cursor_x, temp_image.shape[0]), line_color, 1, cv2.LINE_AA)
    cv2.line(temp_image, (0, cursor_y), (temp_image.shape[1], cursor_y), line_color, 1, cv2.LINE_AA)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_min, y_min = min(ix, x), min(iy, y)
            x_max, y_max = max(ix, x), max(iy, y)
            color = BOX_COLORS[len(bounding_boxes) % len(BOX_COLORS)]
            cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), color, 1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, y_min = min(ix, x), min(iy, y)
        x_max, y_max = max(ix, x), max(iy, y)
        w, h = x_max - x_min, y_max - y_min
        
        bounding_boxes.append({
            "image_id": image_id,
            "bbox": [x_min, y_min, w, h],
            "category_id": selected_category,  # 선택된 패턴 적용
            "id": annotation_id
        })
    
    # 기존 바운딩 박스 다시 그리기 (각각 다른 색상 적용)
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box["bbox"]
        category_id = box["category_id"]
        color = BOX_COLORS[i % len(BOX_COLORS)]
        cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 1)
        label = "Checked" if category_id == 1 else "Solid"
        cv2.putText(temp_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Image", temp_image)

def save_annotations(filename):
    global data, bounding_boxes, image_id, annotation_id, save_count
    
    if filename not in {img["file_name"] for img in data["images"]}:
        image_data = {
            "file_name": filename,
            "height": current_image.shape[0],
            "width": current_image.shape[1],
            "id": image_id
        }
        data["images"].append(image_data)
    
    for box in bounding_boxes:
        box["id"] = annotation_id
        annotation_id += 1
    
    data["annotations"].extend(bounding_boxes)
    
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(data, f, indent=4)
    
    bounding_boxes.clear()
    image_id += 1
    save_count += 1  # 저장된 이미지 카운트 증가


def process_images(image_folder):
    global current_image, backup_image, bounding_boxes, selected_category
    
    existing_files = {img["file_name"] for img in data["images"]}

    for filename in os.listdir(image_folder):
        if filename in existing_files:
            continue
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            
            current_image = image.copy()
            backup_image = image.copy()
            cv2.imshow("Image", current_image)

            # 현재 진행 상황 표시
            total_images = len([f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png") and f not in {img["file_name"] for img in data["images"]}])
            progress_text = f"{save_count}/{total_images}"
            cv2.putText(current_image, progress_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 화면 좌측에 키보드 입력 설명 추가 (항상 표시)
            cv2.putText(current_image, "s: save and next", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(current_image, "n: next (not save)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(current_image, "d: delete and next", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(current_image, "u: undo last box", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(current_image, "q: quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            category_text = "Checked" if selected_category == 1 else "Solid"
            cv2.putText(current_image, f"{category_text}", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.setMouseCallback("Image", draw_rectangle)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    save_annotations(filename)
                    break
                elif key == ord('n'):
                    bounding_boxes.clear()
                    break
                elif key == ord('d'):
                    os.remove(image_path)
                    bounding_boxes.clear()
                    break
                elif key == ord('c'):
                    selected_category = 1  # 체크무늬
                elif key == ord('i'):
                    selected_category = 2  # 단색
                elif key == ord('u') and bounding_boxes:
                    bounding_boxes.pop()
                    draw_rectangle(cv2.EVENT_MOUSEMOVE, cursor_x, cursor_y, None, None)
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    with open(ANNOTATION_FILE, "w") as f:
                        json.dump(data, f, indent=4)
                    exit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_images(image_folder)
