import os
import sys
import cv2
import json
import time
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import threading
import numpy as np

# 현재 src 폴더 상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sys.path 설정 개선
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from detect.region_counter import count_heads_in_regions, load_regions

ava = 0.5

# 전역 변수로 프레임 공유
latest_webcam_frame = None
latest_infer_frame = None
latest_counts = []
latest_timestamp = ""

def save_results_json(counts, filepath="shared_data.json"):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": counts
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

def run_detection(
    model_path="models/best.pt",
    source=0,
    region_cfg_path="configs/regions.yaml",
    interval=ava,
    result_filepath="shared_data.json"
):
    global latest_webcam_frame, latest_infer_frame, latest_counts, latest_timestamp

    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 30

        if not cap.isOpened():
            print("영상 소스를 열 수 없습니다. (웹캠/비디오 없음)")
            latest_webcam_frame = None
            latest_infer_frame = None
            return

        regions = load_regions(region_cfg_path)
        last_run = time.time() - interval
        counts = [0] * len(regions)
        bboxes = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("프레임을 읽지 못했습니다.")
                break

            latest_webcam_frame = frame.copy()
            current_time = time.time()

            if current_time - last_run >= interval:
                results = model(frame, verbose=False)[0]
                new_bboxes = []
                for box in results.boxes:
                    cls_id = int(box.cls)
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        new_bboxes.append((x1, y1, x2, y2))
                bboxes = new_bboxes
                counts = count_heads_in_regions(bboxes, frame.shape, region_cfg_path)
                save_results_json(counts, result_filepath)
                latest_counts = counts
                latest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_run = current_time

            infer_frame = frame.copy()
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region
                cv2.rectangle(infer_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(infer_frame, f"Region {i+1}: {counts[i]}", (x1+5, y1+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            for bbox in bboxes:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(infer_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(infer_frame, "head", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            latest_infer_frame = infer_frame

            cv2.imshow("Head Detection", infer_frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[감지 스레드 예외] {e}")

# FastAPI 앱 정의 (항상 모듈 최상단에서 정의)
app = FastAPI()

@app.get("/")
def serve_html():
    html_path = os.path.abspath(os.path.join(BASE_DIR, 'crowd_status.html'))
    if not os.path.exists(html_path):
        return JSONResponse(content={"error": "crowd_status.html not found"}, status_code=404)
    return FileResponse(html_path, media_type='text/html')

def gen_webcam():
    global latest_webcam_frame
    blank = 255 * np.ones((240, 320, 3), dtype=np.uint8)
    cv2.putText(blank, "No Camera", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    while True:
        frame = latest_webcam_frame if latest_webcam_frame is not None else blank
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.1)

def gen_infer():
    global latest_infer_frame
    blank = 255 * np.ones((240, 320, 3), dtype=np.uint8)
    cv2.putText(blank, "No Inference", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    while True:
        frame = latest_infer_frame if latest_infer_frame is not None else blank
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.get("/webcam_stream")
def webcam_stream():
    return StreamingResponse(gen_webcam(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/inference_stream")
def inference_stream():
    return StreamingResponse(gen_infer(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/data")
def get_data():
    global latest_counts, latest_timestamp
    data = [
        {"region_id": i+1, "head_count": cnt}
        for i, cnt in enumerate(latest_counts)
    ]
    return JSONResponse(content={"timestamp": latest_timestamp, "regions": data})

def start_detection_thread(args):
    t = threading.Thread(target=run_detection, kwargs=dict(
        model_path=args.model,
        source=int(args.source) if str(args.source).isdigit() else args.source,
        region_cfg_path=args.region_cfg,
        interval=args.interval,
        result_filepath=args.output
    ), daemon=True)
    t.start()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/best.pt", help='Path to trained .pt model')
    parser.add_argument('--source', type=str, default='0', help='Webcam index or video file path')
    parser.add_argument('--region_cfg', type=str, default='configs/regions.yaml', help='Region config yaml path')
    parser.add_argument('--interval', type=float, default=ava, help='Detection interval in seconds')
    parser.add_argument('--output', type=str, default="shared_data.json", help='Output result JSON path')
    parser.add_argument('--web', action='store_true', help='Run FastAPI web server')
    args = parser.parse_args()

    if args.web:
        start_detection_thread(args)
        import uvicorn
        uvicorn.run("src.detect.detect_template_web:app", host="0.0.0.0", port=8000, reload=False)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        run_detection(
            model_path=args.model,
            source=src,
            region_cfg_path=args.region_cfg,
            interval=args.interval,
            result_filepath=args.output
        )
