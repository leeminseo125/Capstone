import cv2

# 영상 파일 읽기
cap = cv2.VideoCapture(0)

# 프레임 읽기
ret, frame = cap.read()

# 영상 크기 확인
if ret:
    height, width, channels = frame.shape
    print(f"영상 크기: {width} x {height}")
else:
    print("영상 파일을 읽지 못했습니다.")

# 영상 객체 해제
cap.release()