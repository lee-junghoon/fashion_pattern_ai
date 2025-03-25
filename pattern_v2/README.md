../dataset 폴더에 바운딩 박스 대상 이미지를 넣어둡니다.

bounding.py를 실행 해서 바운딩 박스 작업을 합니다.

작업한 바운딩 박스 정보는 annotation.json 파일에 기록 됩니다.

기본 설정 정보는 bounding.json 파일에 저장 됩니다.

바운딩 작업이 끝나면 ssd_train.py를 실행해 트레이닝 합니다.

트레이닝이 끝나면 테스트할 파일을 test폴더에 넣어두고 test.py를 실행해 테스트 합니다.

테스트가 끝나면 result폴더에 테스트한 파일이 저장 됩니다.

-- API 

api/inference_server.py
START: python inference_server.py --start
STOP: python inference_server.py --stop