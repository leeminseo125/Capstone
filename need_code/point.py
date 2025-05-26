import cv2
import numpy as np
import yaml
import os

video_path = 'test.mp4'
save_dir = './'  # 저장할 디렉토리
os.makedirs(save_dir, exist_ok=True)
yaml_path = os.path.join(save_dir, 'areas.txt')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("영상 파일을 열 수 없습니다.")
    exit()

img = frame.copy()
polygons = []
current_polygon = []

def mouse_callback(event, x, y, flags, param):
    global current_polygon, img
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Frame", img)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

print("다각형의 꼭짓점을 마우스로 클릭하세요.")
print("영역 지정 후 'c' 키를 누르면 다각형이 확정됩니다.")
print("원하는 만큼 다각형을 지정한 뒤 ESC를 누르면 좌표가 txt로 저장됩니다.")

while True:
    temp_img = img.copy()
    if len(current_polygon) > 1:
        cv2.polylines(temp_img, [np.array(current_polygon, np.int32)], isClosed=False, color=(255, 0, 0), thickness=1)
    cv2.imshow("Frame", temp_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(current_polygon) >= 3:
            polygons.append(current_polygon.copy())
            pts = np.array(current_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            # 다각형 중앙에 번호 표시
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img, str(len(polygons)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            current_polygon.clear()
        else:
            print("3개 이상의 점을 찍어야 다각형이 됩니다.")
    if key == 27:
        break

cv2.destroyAllWindows()

# yaml로 저장
areas = {}
for idx, poly in enumerate(polygons):
    areas[f'area_{idx+1}'] = [ [int(x), int(y)] for (x, y) in poly ]

# Python 리스트 형식으로 저장 (지역 이름: region_1, region_2, ...)
with open(yaml_path, 'w', encoding='utf-8') as f:
    for poly in polygons:
        coords = [tuple(map(int, pt)) for pt in poly]
        f.write(f"{coords},\n")

print(f"영역 좌표가 {yaml_path}에 저장되었습니다.")

print(f"영역 좌표가 {yaml_path}에 Python 리스트 형식으로 저장되었습니다.")
