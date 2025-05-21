import os
import sys
import cv2
import json
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import streamlit as st

# 현재 src 폴더 상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect.region_counter import count_heads_in_regions, load_regions

# Streamlit, torch, ultralytics 등에서 발생하는 모듈 경로 추적 오류 방지용
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Streamlit 핫리로딩/torch 경로 추적 오류 방지용
import sys
if "streamlit" in sys.modules:
    import importlib
    import types
    sys.modules["torch.__path__"] = types.SimpleNamespace(_path=[])

ava = 0.5

def save_results_json(counts, regions, filepath="result.json"):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regions": [
            {"region_id": i+1, "head_count": cnt}
            for i, cnt in enumerate(counts)
        ]
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def run_detection_streamlit(
    model_path="models/best.pt",
    source=0,
    region_cfg_path="configs/regions.yaml",
    interval=ava
):
    # torch/ultralytics/streamlit watcher 관련 경로 추적 오류 방지
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO 모델 로드 실패: {e}")
        return

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.03

    if not cap.isOpened():
        st.error("영상 소스를 열 수 없습니다. (웹캠/비디오 없음)")
        return

    try:
        regions = load_regions(region_cfg_path)
    except Exception as e:
        st.error(f"구역 설정 파일 로드 실패: {e}")
        return

    last_run = time.time() - interval
    counts = [0] * len(regions)
    bboxes = []

    stframe_webcam = st.empty()
    stframe_infer = st.empty()
    sttable = st.empty()

    # 종료 버튼을 루프 밖에서 한 번만 생성
    stop = st.button("종료", key="stop")

    while cap.isOpened():
        if stop:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("프레임을 읽지 못했습니다.")
            break

        # interval마다 추론
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
            last_run = current_time
            # 결과를 실시간으로 result.json에 저장
            save_results_json(counts, regions, filepath="result.json")

        # 추론 결과 프레임 생성
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

        # Streamlit에 표시
        stframe_webcam.image(frame[..., ::-1], channels="RGB", caption="원본 영상")
        stframe_infer.image(infer_frame[..., ::-1], channels="RGB", caption="추론 결과")
        sttable.table(
            [{"구역 번호": i+1, "탐지 인원수": cnt} for i, cnt in enumerate(counts)]
        )

        time.sleep(frame_delay)

    cap.release()

def main():
    st.title("셔틀버스 혼잡도 실시간 추론 (Streamlit)")
    st.sidebar.header("설정")
    model_path = st.sidebar.text_input("모델 경로", "models/best.pt")
    region_cfg_path = st.sidebar.text_input("구역 설정 YAML", "configs/regions.yaml")
    interval = st.sidebar.number_input("추론 주기(초)", min_value=0.1, value=0.5, step=0.1)
    source_type = st.sidebar.selectbox("소스 타입", ["웹캠", "비디오 파일"])
    if source_type == "웹캠":
        source = 0
    else:
        source = st.sidebar.text_input("비디오 파일 경로", "test.mp4")

    if st.button("실시간 추론 시작"):
        run_detection_streamlit(
            model_path=model_path,
            source=source,
            region_cfg_path=region_cfg_path,
            interval=interval
        )

if __name__ == "__main__":
    main()
