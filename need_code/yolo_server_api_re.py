from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

areas_path = './areas.txt'

# 전역 변수
areas = []
regions = {}

# ✅ 영역 파일을 읽어오는 함수
def load_areas():
    global areas, regions
    try:
        new_areas = []
        with open(areas_path, 'r', encoding='utf-8') as f:
            for line in f:
                coords = eval(line.strip().rstrip(','))
                new_areas.append(coords)
        areas = new_areas

        # Polygon 갱신
        new_regions = {}
        for i, area in enumerate(areas):
            new_regions[f"region-{i+1:02}"] = Polygon(area)
        regions = new_regions

        print("✅ areas.txt 파일이 다시 로드되었습니다.")
    except Exception as e:
        print(f"❌ areas.txt 로딩 중 오류: {e}")

# ✅ 처음 한 번 로드
load_areas()

# ✅ Watchdog 이벤트 핸들러
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("areas.txt"):
            print("📂 areas.txt 변경 감지됨, 다시 로드 중...")
            load_areas()

# ✅ Watchdog 감시 스레드 시작
def start_file_watcher():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(areas_path)), recursive=False)
    observer.start()

# 백그라운드 스레드로 실행
threading.Thread(target=start_file_watcher, daemon=True).start()

# ✅ YOLO 모델 로드
model = YOLO("best.pt")

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

        results = model(img)[0]
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
