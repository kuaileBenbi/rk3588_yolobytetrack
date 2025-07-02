from collections import deque
import time
from visionar import VisionTracker
import sys
import cv2


# 用来保存每帧的起始时间戳： key=frame_id, value=start_time
start_times = {}
end_time = {}
# 滑动窗口记录最近 N 帧的回调时间戳，计算 FPS
callback_times = deque(maxlen=100)

# 全局自增帧 ID
_frame_id = 0


def handle_result(results, ret, frame_id):
    """
    external_callback: 每当有新结果时被调用
    args:
      - results: VisionTracker 返回的结果 dict
      - frame_id: 我们在 put() 时传给它的 ID
    """
    now = time.time()
    # 计算这帧的延迟
    t0 = start_times.pop(frame_id, None)
    if t0 is not None:
        latency_ms = (now - t0) * 1000
    else:
        latency_ms = -1  # 理论上不该发生

    # 更新滑动窗口
    callback_times.append(now)

    # 打印当前这帧的延迟 & FPS
    # FPS  = 窗口中帧数 / (最早回调到现在的时间间隔)
    fps = (
        len(callback_times) / (now - callback_times[0])
        if len(callback_times) > 1
        else 0.0
    )

    print(f"[Frame {frame_id}] Latency: {latency_ms:.1f} ms \t FPS: {fps:.1f}")
    # print(f"ret: {ret}")
    # cv2.imwrite(f"detected_{frame_id}.jpg", results["frame"])


if __name__ == "__main__":

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "视频打开失败"

    vt = VisionTracker(TPEs=3, external_callback=handle_result)
    vt.set_tracking_mode(False)

    _frame_id = 0
    start_time = time.time()
    total_frames = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        fid = _frame_id
        start_times[fid] = time.time()
        # s_t = time.time()
        vt.detworking(frame, frame_id=fid)
        # e_t = time.time()
        # print(f"放队列时间消耗: {(e_t-s_t):.2f} seconds")

        # end_times[fid] = time.time()
        # total_frames += 1
        _frame_id += 1
        time.sleep(0.03)

    cap.release()
    vt.stop_detworking()

    # 计算总时间和平均帧频
    # elapsed_time = time.time() - start_time
    # average_fps = total_frames / elapsed_time if elapsed_time > 0 else 0

    # print(f"Total frames read: {total_frames}")
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # print(f"Average FPS: {average_fps:.2f}")
