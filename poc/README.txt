필요한 파이썬 패키지
opencv-python
numpy
torch
torchvision
Pillow

bounding_box_coco.py 바운딩박스 그리는 툴
image_regulation.py 이미지 300x300으로 리사이징 하는 코드
ssd_train.py ssd 트레이닝 코드
test.py 모델 검증


순서
1. 준비된 파일을 image_regulation.py을 실행해서 리사이즈 합니다.
2. bounding_box_coco.py 실행해서 바운딩 박스를 그립니다.
3. 2의 작업이 끝나면 ssd_train.py 실행해서 학습을 진행 합니다.
4. test.py를 실행해서 학습한 모델로 추론을 합니다.


