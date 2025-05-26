from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

areas = []
with open('areas.txt', 'r', encoding='utf-8') as f:
    for line in f:
        coords = eval(line.strip().rstrip(','))
        areas.append(coords)

# 영역 정의
region_1 = areas[0]
region_2 = areas[1]

regions = {
    "region-01": Polygon(region_1),
    "region-02": Polygon(region_2)
}

# YOLO 모델 로드
model = YOLO("best.pt")  # 커스텀 모델 경로

@app.get("/")
async def root():
    return {"message": "✅ YOLO Detection 기반 Region 카운팅 API 입니다."}

@app.get("/predict")
async def get_predict_info():
    return {
        "message": "✅ YOLO Detection 기반 Region 카운팅 API입니다.",
        "how_to_use": "POST /predict with form-data: frame=<image_file>"
    }

@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    try:
        file_bytes = await frame.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "이미지 디코딩 실패"})

        results = model(img)[0]  # 첫 번째 결과
        region_counts = {key: 0 for key in regions.keys()}

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            center_point = Point(cx, cy)

            for region_name, polygon in regions.items():
                if polygon.contains(center_point):
                    region_counts[region_name] += 1

            # 바운딩 박스 및 중심점 표시
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        # 영역 폴리곤 그리기
        for polygon in regions.values():
            pts = np.array(polygon.exterior.coords, np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "region_counts": region_counts,
            "result_image": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
