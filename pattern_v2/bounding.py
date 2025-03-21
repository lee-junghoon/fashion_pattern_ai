import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json

# 설정 저장 파일
CONFIG_FILE = "bounding.json"
ANNOTATION_FILE = "annotation.json"

# 사용할 색상 7가지 (로테이션)
BOX_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
color_index = 0

CATEGORY_MAPPING = {"checked": 1, "solid": 2, "stripe": 3, "dotted": 4, "floral": 5, "animal": 6, "paisley": 7, "printed": 8 }  # category_id 매핑 (COCO 포맷용)
current_category = "checked"  # 기본 카테고리

def load_config():
    global current_image_index
    if os.path.exists("bounding.json"):
        with open("bounding.json", "r") as f:
            try:
                config = json.load(f)
                entry.insert(0, config.get("last_directory", ""))
                save_entry.insert(0, config.get("last_save_directory", ""))
                current_image_index = config.get("current_index", 0)
            except Exception as e:
                print("Failed to load bounding.json:", e)
    return config

def save_config():
    config = {
        "last_directory": entry.get(),
        "last_save_directory": save_entry.get(),
        "current_index": current_image_index
    }
    with open("bounding.json", "w") as f:
        json.dump(config, f, indent=2)

def on_closing():
    save_config()
    root.destroy()

config = load_config()

root = tk.Tk()
root.title("Bounding Box Drawer")
root.geometry("1000x600")
root.protocol("WM_DELETE_WINDOW", on_closing)

left_frame = tk.Frame(root, width=400, height=600, bg="lightgray", padx=10, pady=10)
left_frame.pack(side="left", fill="both")
right_frame = tk.Frame(root, width=600, height=600, bg="white")
right_frame.pack(side="right", fill="both", expand=True)

canvas = tk.Canvas(right_frame, width=600, height=600, bg="white", cursor="none")
canvas.pack()

input_frame = tk.Frame(left_frame, bg="lightgray")
input_frame.pack(pady=10, fill="x")

label = tk.Label(input_frame, text="Directory:", bg="lightgray")
label.pack(side="left", padx=5)

entry = tk.Entry(input_frame, width=30)
entry.insert(0, config["last_directory"])
entry.pack(side="left", padx=5)

button = tk.Button(input_frame, text="Load Image", command=lambda: load_image())
button.pack(side="left", padx=5)

save_frame = tk.Frame(left_frame, bg="lightgray")
save_frame.pack(pady=10, fill="x")

save_label = tk.Label(save_frame, text="Save To:", bg="lightgray")
save_label.pack(side="left", padx=5)

save_entry = tk.Entry(save_frame, width=30)
save_entry.insert(0, config["last_save_directory"])
save_entry.pack(side="left", padx=5)

# 하단 버튼 프레임 (Exit + Help)
bottom_frame = tk.Frame(left_frame, bg="lightgray")
bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

def show_help():
    messagebox.showinfo("도움말", 
        "Directory: 이미지가 있는 디렉토리 경로\n"
        "Save To: 바운딩 박스 처리한 이미지가 저장되는 경로\n"
        "Load Image를 눌러 이미지를 불러 오세요\n"
        "\n"
        "패턴을 선택하고 드래그로 바운딩 박스를 그리세요\n"
        "마우스 휠로 확대/축소 할 수 있어요.\n"
        "\n"
        "단축키:\n"
        "- Ctrl + S : 저장 & 다음 이미지 & \n"
        "- Ctrl + Z : 바운딩 박스 실행 취소\n"
        "- Ctrl + Backspace : 이전 이미지\n\n"
        "Tip:\n"
        "- Save To 경로에 300x300으로 리사이즈된 이미지가 저장됩니다\n"
        "- 라벨링 정보는 COCO포맷으로 annotation.json 파일에 저장 됩니다.\n"
    )
help_button = tk.Button(bottom_frame, text="Help", command=show_help, bg="#5bc0de", fg="white", padx=10, pady=5)
help_button.pack(side="left", fill="x", padx=10)

exit_button = tk.Button(bottom_frame, text="Quit", command=root.quit, bg="#d9534f", fg="white", padx=10, pady=5)
exit_button.pack(side="left", expand=True, fill="x", padx=(5, 0))

# Next/Back 버튼 추가
nav_frame = tk.Frame(left_frame, bg="lightgray")
nav_frame.pack(pady=10)
save_anno_button = tk.Button(nav_frame, text="Save and Next Image (Ctrl + s)", command=lambda: save_annotations())
save_anno_button.pack(pady=5)
next_button = tk.Button(nav_frame, text="Skip and Next Image (Ctrl + x)", command=lambda: navigate_image(1))
next_button.pack(padx=5)
back_button = tk.Button(nav_frame, text="Back Image (Ctrl + Backspace)", command=lambda: navigate_image(-1))
back_button.pack(padx=5)
undo_button = tk.Button(nav_frame, text="Undo (Ctrl + z)", command=lambda: undo_bounding_box())
undo_button.pack(pady=5)

category_var = tk.StringVar(value="checked")
category_frame = tk.LabelFrame(left_frame, text="Patterns", bg="lightgray", padx=5, pady=5)
category_frame.pack(pady=5)

for category in CATEGORY_MAPPING.keys():
    radio = tk.Radiobutton(
        category_frame,
        text=category.capitalize(),
        variable=category_var,
        value=category,
        bg="lightgray",
        anchor="w"
    )
    radio.pack(anchor="w")

selected_image = None
current_img = None
scale_factor = 1.0
photo_img = None
bounding_boxes = []  # (rel_x1, rel_y1, rel_x2, rel_y2, color, category)
start_x, start_y = None, None
cursor_lines = []

progress_label = tk.Label(left_frame, text="Progress: 0 / 0", bg="lightgray", font=("Arial", 10))
progress_label.pack(pady=5)

def update_progress_label():
    if image_list:
        progress_text = f"Progress: {current_image_index + 1} / {len(image_list)}"
    else:
        progress_text = "Progress: 0 / 0"
    progress_label.config(text=progress_text)

current_image_index = 0
image_list = []
annotation_data = {}

def load_image_by_index(index):
    global selected_image, current_img, scale_factor, bounding_boxes
    dir_path = entry.get()
    if not image_list:
        load_image_list()
    if not image_list or index >= len(image_list):
        return
    image_name = image_list[index]
    img_path = os.path.join(dir_path, image_name)
    selected_image = img_path
    img = Image.open(img_path).resize((300, 300))
    current_img = img
    scale_factor = 1.0
    bounding_boxes.clear()
    update_image()
    update_progress_label()

def navigate_image(step):
    global current_image_index, bounding_boxes
    if not image_list:
        load_image_list()
    new_index = current_image_index + step
    if 0 <= new_index < len(image_list):
        current_image_index = new_index
        load_image_by_index(current_image_index)
        update_progress_label()

def load_image_list():
    global image_list, current_image_index
    dir_path = entry.get()
    if os.path.isdir(dir_path):
        image_list = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", "jpg", "jpeg"))]
        image_list.sort()
    
    # 마지막 작업한 이미지 다음부터 시작
    last_image = None
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, "r") as f:
            try:
                annotation_data = json.load(f)
                if annotation_data["images"]:
                    last_image = annotation_data["images"][-1]["file_name"]
            except json.JSONDecodeError:
                pass
    
    # 인덱스 찾아서 다음 이미지로 설정
        if last_image and last_image in image_list:
            last_index = image_list.index(last_image)
            if last_index + 1 < len(image_list):
                current_image_index = last_index + 1
            else:
                current_image_index = last_index  # 마지막 이미지일 경우 그 이미지로 유지
        else:
            current_image_index = 0
        

def load_image():
    global selected_image, current_img, scale_factor, bounding_boxes
    dir_path = entry.get()
    save_config()
    if os.path.isdir(dir_path):
        images = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", "jpg", "jpeg"))]
        if images:
            img_path = os.path.join(dir_path, images[0])
            selected_image = img_path
            img = Image.open(img_path)
            img = img.resize((300, 300))
            current_img = img
            scale_factor = 1.0
            bounding_boxes.clear()
            update_image()

def update_image():
    global photo_img
    if current_img:
        new_size = int(300 * scale_factor)
        new_size = min(max(new_size, 300), 600)
        resized_img = current_img.resize((new_size, new_size))
        photo_img = ImageTk.PhotoImage(resized_img)
        canvas.delete("all")
        center_x = (600 - new_size) // 2
        center_y = (600 - new_size) // 2
        canvas.create_image(center_x, center_y, anchor=tk.NW, image=photo_img)

        # 이미지 영역 회색 테두리
        canvas.create_rectangle(center_x, center_y, center_x + new_size, center_y + new_size, outline="gray", width=1)
        
        for rx1, ry1, rx2, ry2, color, category in bounding_boxes:
            x1 = center_x + rx1 * new_size
            y1 = center_y + ry1 * new_size
            x2 = center_x + rx2 * new_size
            y2 = center_y + ry2 * new_size
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="bounding_box")
            canvas.create_text(x1 + 3, y1, text=category, anchor="nw", fill=color, font=("Arial", 10, "bold"))

def zoom(event):
    global scale_factor
    if event.widget == canvas:
        if event.delta > 0 and (300 * scale_factor * 1.1) <= 600:
            scale_factor *= 1.1
        elif event.delta < 0 and (300 * scale_factor / 1.1) >= 300:
            scale_factor /= 1.1
        update_image()

def start_draw(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

def draw_rectangle(event): 
    global color_index
    if start_x and start_y:
        canvas.delete("current_box")

        # 현재 색상 가져오기 (박스 개수 기준이 아니라 현재 인덱스 기준)
        current_color = BOX_COLORS[color_index % len(BOX_COLORS)]

        # 사각형 실시간 미리보기 (현재 색상으로)
        canvas.create_rectangle(
            start_x, start_y, event.x, event.y,
            outline=current_color,
            width=2,
            tags="current_box"
        )
def finalize_rectangle(event):
    global color_index
    if start_x and start_y:
        new_size = int(300 * scale_factor)
        new_size = min(max(new_size, 300), 600)
        center_x = (600 - new_size) // 2
        center_y = (600 - new_size) // 2
        rel_x1 = (start_x - center_x) / new_size
        rel_y1 = (start_y - center_y) / new_size
        rel_x2 = (event.x - center_x) / new_size
        rel_y2 = (event.y - center_y) / new_size
        color = BOX_COLORS[color_index % len(BOX_COLORS)]
        color_index += 1
        category = category_var.get()
        bounding_boxes.append((rel_x1, rel_y1, rel_x2, rel_y2, color, category))
        update_image()

def undo_bounding_box():
    if bounding_boxes:
        bounding_boxes.pop()
        update_image()

def show_crosshair(event):
    global cursor_lines
    for line in cursor_lines:
        canvas.delete(line)
    cursor_lines.clear()
    x, y = event.x, event.y
    hline = canvas.create_line(0, y, 600, y, fill="gray", width=1)
    vline = canvas.create_line(x, 0, x, 600, fill="gray", width=1)
    cursor_lines.extend([hline, vline])

def clear_crosshair(event):
    for line in cursor_lines:
        canvas.delete(line)
    cursor_lines.clear()

def save_annotations():
    if not selected_image:
        messagebox.showwarning("No Image", "Load an image first.")
        return
    image_id = os.path.basename(selected_image)
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, "r") as f:
            annotation_data = json.load(f)
    else:
        annotation_data = {
            "images": [],
            "categories": [
                {"id": 1, "name": "checked"},
                {"id": 2, "name": "solid"}
            ],
            "annotations": []
        }

    # Remove previous annotations for this image
    annotation_data["annotations"] = [ann for ann in annotation_data["annotations"] if ann["image_id"] != image_id]
    annotation_data["images"] = [img for img in annotation_data["images"] if img["id"] != image_id]

    annotation_data["images"].append({"id": image_id, "file_name": image_id, "width": 300, "height": 300})
    for i, (rx1, ry1, rx2, ry2, _, category) in enumerate(bounding_boxes):
        x = min(rx1, rx2) * 300
        y = min(ry1, ry2) * 300
        w = abs(rx2 - rx1) * 300
        h = abs(ry2 - ry1) * 300
        annotation_data["annotations"].append({
            "id": len(annotation_data["annotations"]) + 1,
            "image_id": image_id,
            "category_id": CATEGORY_MAPPING[category],
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0
        })

    with open(ANNOTATION_FILE, "w") as f:
        json.dump(annotation_data, f, indent=2)

        # Save resized image to save directory
        save_dir = save_entry.get()

        if os.path.isdir(save_dir):
            resized_img = current_img.resize((300, 300))
            file_name = os.path.basename(selected_image)
            save_path = os.path.join(save_dir, file_name)
            try:
                resized_img.save(save_path)
            except Exception as e:
                print(f"Failed to save resized image: {e}")

    navigate_image(1)  

canvas.bind("<MouseWheel>", zoom)
canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<B1-Motion>", draw_rectangle)
canvas.bind("<ButtonRelease-1>", finalize_rectangle)
canvas.bind("<Motion>", show_crosshair)
canvas.bind("<Leave>", clear_crosshair)

if config["last_directory"]:
    load_image_list()
    load_image_by_index(current_image_index)
    update_progress_label()


root.bind("<Control-z>", lambda event: undo_bounding_box())
root.bind("<Control-s>", lambda event: save_annotations())
root.bind("<Control-x>", lambda event: navigate_image(1))
root.bind("<Control-BackSpace>", lambda event: navigate_image(-1))

root.mainloop()
