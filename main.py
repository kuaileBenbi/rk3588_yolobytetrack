from visionar import VisionTracker
import sys
import cv2


def handle_result(results, ret):
    # 每当有新结果时，就会调用这里
    print("检测到结果: ", ret)
    cv2.imwrite("test.jpg", results["frame"])


video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "视频打开失败"

vt = VisionTracker(TPEs=3, external_callback=handle_result)
vt.set_tracking_mode(True)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    vt.detworking(frame)

cap.release()
vt.stop_detworking()
