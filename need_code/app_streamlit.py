import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import base64
import time

st.set_page_config(page_title="YOLO Region Stream", layout="wide")
st.title("🧠 YOLO 영역 기반 실시간 스트리밍")

ORIGINAL_VIDEO_URL = 'http://127.0.0.1:8080/video_feed'
YOLO_PREDICT_URL = 'http://127.0.0.1:8081/predict'

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
        res.raise_for_status()  # HTTP 오류가 있으면 예외 발생
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

def main():
    original_stream = stream_frames(ORIGINAL_VIDEO_URL)

    last_update = 0
    region_counts = {}

    for orig_frame in original_stream:
        if orig_frame is None:
            continue

        original_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        original_placeholder.image(Image.fromarray(original_rgb), use_container_width=True)

        now = time.time()
        if now - last_update > 0.5:
            region_counts, detected_img = predict_regions(orig_frame)
            last_update = now

            if detected_img is not None:
                detected_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                detected_placeholder.image(Image.fromarray(detected_rgb), use_container_width=True)

            region1_placeholder.markdown(f"### 🟢 Region-01 Count: `{region_counts.get('region-01', 0)}`")
            region2_placeholder.markdown(f"### 🔵 Region-02 Count: `{region_counts.get('region-02', 0)}`")

if __name__ == "__main__":
    main()
