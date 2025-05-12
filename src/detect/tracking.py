import os
import sys
import cv2
import json
import time
from datetime import datetime
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect.region_counter import load_regions, is_inside_region, box_center

# 결과 JSON 저장 함수
def save_results_json(counts, filepath="shared_data.json"):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": counts
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

# 추론 및 Tracking 함수 (보완 버전)
def run_tracking(
    model_path="models/best.pt",
    source="./videos/track_test.mp4",
    region_cfg_path="configs/regions.yaml",
    interval=5,
    track_frame_interval=5,  # Tracking을 몇 프레임마다 수행할지 설정
    result_filepath="shared_data.json"
):
    model = YOLO(model_path)
    regions = load_regions(region_cfg_path)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)

    if not cap.isOpened():
        raise RuntimeError("영상 소스를 열 수 없습니다.")

    last_run = time.time() - interval
    counts = [0] * len(regions)
    frame_count = 0
    bboxes = []  # 항상 초기화해두기

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다.")
            break

        current_time = time.time()
        frame_count += 1

        if frame_count % track_frame_interval == 0:
            # YOLO + ByteTrack Tracking 수행
            results = model.track(frame, persist=True, verbose=False)[0]

            # 디버깅용 Tracking 정보 출력
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().tolist()
                bboxes = results.boxes.xyxy.int().tolist()
                cls_ids = results.boxes.cls.int().tolist()

                # print(f"\n[Tracking Debug Info]")
                # print(f"Tracking IDs: {track_ids}")
                # print(f"Class IDs: {cls_ids}")
                # print(f"Bounding Boxes: {bboxes}")
            else:
                print("\n[Tracking Debug Info] No tracking IDs assigned.")
                track_ids, bboxes, cls_ids = [], [], []

            id_to_region = {}
            counts = [0] * len(regions)

            for track_id, cls_id, box in zip(track_ids, cls_ids, bboxes):
                # 클래스 ID 반드시 확인하고 수정 필요!
                if cls_id == 0:  # 클래스 ID 확인 필수
                    center = box_center(*box)
                    for i, region in enumerate(regions):
                        if is_inside_region(center, region):
                            id_to_region[track_id] = i
                            break

            unique_regions = set(id_to_region.values())
            for region_id in unique_regions:
                counts[region_id] = list(id_to_region.values()).count(region_id)

            # 지정된 interval 초마다 결과 저장
            if current_time - last_run >= interval:
                save_results_json(counts, result_filepath)
                print(f"[{datetime.now()}] 결과 저장됨: {counts}")
                last_run = current_time

        # 매 프레임 시각화
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"Region {i+1}: {counts[i]}", (x1+5,y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        cv2.imshow("Head Tracking", frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 실행 코드
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/best.pt")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--region_cfg', type=str, default='configs/regions.yaml')
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--track_interval', type=int, default=5,
                        help="Number of frames between tracking updates")
    parser.add_argument('--output', type=str, default="shared_data.json")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    run_tracking(
        model_path=args.model,
        source=src,
        region_cfg_path=args.region_cfg,
        interval=args.interval,
        track_frame_interval=args.track_interval,
        result_filepath=args.output
    )
