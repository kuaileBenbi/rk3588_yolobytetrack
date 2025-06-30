from visionar import VisionTracker
import sys
import cv2


def handle_result(result):
    # 每当有新结果时，就会调用这里
    print("检测到结果!")


video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "视频打开失败"

vt = VisionTracker(TPEs=3, callback=handle_result)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    vt.detworking(frame)

cap.release()
vt.stop_detworking()
