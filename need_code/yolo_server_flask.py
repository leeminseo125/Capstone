from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# 영역 정의
region_1 = [(340, 228), (532, 224), (599, 395), (243, 364)]
region_2 = [(395, 66), (374, 192), (568, 201), (647, 420), (802, 413), (764, 69)]

regions = {
    "region-01": Polygon(region_1),
    "region-02": Polygon(region_2)
}

# YOLO 모델 로드
model = YOLO("models/best.pt")  # 너의 커스텀 모델 경로

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "message": "✅ YOLO Detection 기반 Region 카운팅 API입니다.",
            "how_to_use": "POST /predict with form-data: frame=<image_file>"
        })

    if 'frame' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다. 'frame' 필드로 전송해주세요."}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "이미지 디코딩 실패"}), 400

    # 객체 탐지
    results = model(img)[0]  # 첫 번째 결과

    # region별 카운트 초기화
    region_counts = {key: 0 for key in regions.keys()}

    # 결과 처리
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # 중심점

        center_point = Point(cx, cy)
        for region_name, polygon in regions.items():
            if polygon.contains(center_point):
                region_counts[region_name] += 1

        # 바운딩 박스 및 중심점 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    # 영역 폴리곤 그리기
    for polygon in regions.values():
        pts = np.array(polygon.exterior.coords, np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # 이미지 base64 인코딩
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "region_counts": region_counts,
        "result_image": img_base64
    })

@app.route('/')
def index():
    return "✅ YOLO Detection 기반 Region 카운팅 서버 정상 동작 중"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
