import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"

# 영상 크기 확인
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {w}x{h}")  # 확인용

# 1번 영역: 오각형 (5점)
region_1 = [(100, 100), (200, 80), (280, 150), (250, 220), (150, 220)]

# 2번 영역: 사다리꼴 (4점)
region_2 = [(400, 300), (560, 300), (520, 400), (380, 400)]

region_points = {
    "region-01": region_1,
    "region-02": region_2,
}

regioncounter = solutions.RegionCounter(
    show=False,
    region=region_points,
    model="models/best.pt",
)

while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = regioncounter(frame)

    print(results.region_counts)  # 영역별 카운트 출력

    output_img = results.plot_im

    cv2.imshow("Region Counting", output_img)

    if cv2.waitKey(1) == 27:  # ESC키 종료
        break

cap.release()
cv2.destroyAllWindows()
