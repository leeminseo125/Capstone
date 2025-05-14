import os
import sys
import cv2
import json
import time
from datetime import datetime
from ultralytics import YOLO

# 현재 src 폴더 상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect.region_counter import count_heads_in_regions, load_regions

ava = 0.5

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
    interval=ava,  # 기본 추론/저장 주기를 1초로 변경
    result_filepath="shared_data.json"
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture('HT21-04-raw.webm')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)
    
    if not cap.isOpened():
        raise RuntimeError("영상 소스를 열 수 없습니다.")

    regions = load_regions(region_cfg_path)

    last_run = time.time() - interval
    counts = [0] * len(regions)  # 초기 counts를 0으로 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("프레임을 읽지 못했습니다.")
            break

        current_time = time.time()

        # 10초마다 추론
        if current_time - last_run >= interval:
            results = model(frame, verbose=False)[0]

            bboxes = []
            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bboxes.append((x1, y1, x2, y2))

            counts = count_heads_in_regions(bboxes, frame.shape, region_cfg_path)
            save_results_json(counts, result_filepath)
            print(f"{datetime.now()} 결과 저장됨: {counts}")

            last_run = current_time

        # 매 프레임마다 최근 추론 결과 시각화 
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Region {i+1}: {counts[i]}", (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # bbox 시각화 추가
        for bbox in bboxes:
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, "head", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Head Detection", frame)

        # 영상 속도에 맞추기
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/best.pt", help='Path to trained .pt model')
    parser.add_argument('--source', type=str, default='0', help='Webcam index or video file path')
    parser.add_argument('--region_cfg', type=str, default='configs/regions.yaml', help='Region config yaml path')
    parser.add_argument('--interval', type=int, default=ava, help='Detection interval in seconds')  # 기본값 1초로 변경
    parser.add_argument('--output', type=str, default="shared_data.json", help='Output result JSON path')
    args = parser.parse_args()

    # 웹캠 소스인 경우 정수로 변경
    src = int(args.source) if args.source.isdigit() else args.source

    run_detection(
        model_path=args.model,
        source=src,
        region_cfg_path=args.region_cfg,
        interval=args.interval,
        result_filepath=args.output
    )


