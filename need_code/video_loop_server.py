from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import time
import io

app = FastAPI()

video_path = 'test.mp4'  # MOT 영상 경로
cap = cv2.VideoCapture(video_path)


def generate_frames():
    global cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps

    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        time.sleep(delay)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Stream</title>
    </head>
    <body>
        <h1>Streaming from video file (FastAPI)</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_loop_server:app", host="0.0.0.0", port=8080, reload=True)
