from flask import Flask, Response, render_template_string
import cv2
import time

app = Flask(__name__)
video_path = 'test.mp4'  # MOT 영상 경로
cap = cv2.VideoCapture(video_path)

def generate():
    global cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 가져오기 (기본 30)
    delay = 1.0 / fps  # 프레임 간 시간

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            continue

        _, jpeg = cv2.imencode('.jpg', frame)
        time.sleep(delay)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Stream</title>
    </head>
    <body>
        <h1>Streaming from video file</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
