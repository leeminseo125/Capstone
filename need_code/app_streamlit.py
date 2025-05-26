import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import base64
import time
import subprocess
import sys
import os

st.set_page_config(page_title="YOLO Region Stream", layout="wide")
st.title("🧠 YOLO 영역 기반 실시간 스트리밍")

ORIGINAL_VIDEO_URL = 'http://127.0.0.1:8080/video_feed'
YOLO_PREDICT_URL = 'http://127.0.0.1:8000/predict'

col1, col2 = st.columns(2)
col1.title("🎥 원본 영상 (스트리밍 서버)")
col2.title("🧠 YOLO 추론 영상 (추론 서버)")

original_placeholder = col1.empty()
detected_placeholder = col2.empty()

region1_placeholder = st.empty()
region2_placeholder = st.empty()

def stream_frames(url):
    stream = requests.get(url, stream=True)
    bytes_stream = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_stream += chunk
        a = bytes_stream.find(b'\xff\xd8')
        b = bytes_stream.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_stream[a:b+2]
            bytes_stream = bytes_stream[b+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield img

def predict_regions(img):
    _, img_encoded = cv2.imencode('.jpg', img)
    files = {'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        res = requests.post(YOLO_PREDICT_URL, files=files, timeout=5)
        res.raise_for_status()
        data = res.json()

        region_counts = data.get("region_counts", {})
        img_base64 = data.get("result_image", "")

        if img_base64:
            img_bytes = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            detected_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            detected_img = None

        return region_counts, detected_img

    except Exception as e:
        st.error(f"YOLO 서버 요청 실패 또는 응답 문제: {e}")
        return {}, None

def run_area_setting_script():
    # 영역 설정 스크립트 파일명
    script_path = os.path.join(os.path.dirname(__file__), "point.py")
    python_exe = sys.executable
    result = subprocess.run([python_exe, script_path])
    if result.returncode == 0:
        st.success("영역 좌표가 areas.txt에 저장되었습니다.")
    else:
        st.error("영역 설정 도중 오류가 발생했습니다.")

def main():
    if st.button("영역지정"):
        run_area_setting_script()
        # 최신 streamlit(1.18.0 이상)에서는 아래 코드 사용
        try:
            st.experimental_rerun()
        except AttributeError:
            st.warning("영역 설정이 완료되었습니다. 페이지를 새로고침(F5) 해주세요.")
            st.stop()

    original_stream = stream_frames(ORIGINAL_VIDEO_URL)

    last_update = 0
    region_counts = {}

    for orig_frame in original_stream:
        if orig_frame is None:
            continue

        original_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        original_placeholder.image(Image.fromarray(original_rgb))

        now = time.time()
        if now - last_update > 0.5:
            region_counts, detected_img = predict_regions(orig_frame)
            last_update = now

            if detected_img is not None:
                detected_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                detected_placeholder.image(Image.fromarray(detected_rgb))

            region1_placeholder.markdown(f"### 🟢 Region-01 Count: `{region_counts.get('region-01', 0)}`")
            region2_placeholder.markdown(f"### 🔵 Region-02 Count: `{region_counts.get('region-02', 0)}`")

if __name__ == "__main__":
    main()