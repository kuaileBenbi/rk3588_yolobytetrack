from queue import Empty
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


def initRKNN(rknnModel="rknnModel/yolov8n.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, f"\t\tdetect on NPU-{id}")
    return rknn_lite


def initRKNNs(rknnModel="rknnModel/yolov8n.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list


class detectExecutor:
    def __init__(self, det_model, TPEs, func, callback):
        """
        det_model: 模型路径或对象
        TPEs: 并行实例数
        func: 検測函数，签名 func(rknn_instance, frame) -> result
        callback: 结果回调，签名 callback(result: Any)
        """
        self.TPEs = TPEs
        self.func = func
        self.callback = callback

        # 初始化 RKNN 实例池和线程池
        self.rknnPool = initRKNNs(det_model, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self._round_robin = 0

    def put(self, frame):
        """
        提交一个检测任务，计算完成后自动触发 callback(result)。
        """
        idx = self._round_robin
        rknn_ins = self.rknnPool[idx]
        self._round_robin = (idx + 1) % self.TPEs

        fut = self.pool.submit(self.func, rknn_ins, frame)
        # 给 future 添加回调
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut):
        """
        内部回调，获取 future 结果并调用用户回调。
        """
        try:
            result = fut.result()
            # 将结果传递给外部回调
            self.callback(result)
        except Exception as e:
            # 这里也可以把异常通过 callback 传出去，或另行日志/处理
            print(f"[detectExecutor] 任务执行异常: {e}")

    def release(self):
        """
        关停线程池和释放 RKNN 资源
        """
        self.pool.shutdown(wait=True)
        for inst in self.rknnPool:
            inst.release()
