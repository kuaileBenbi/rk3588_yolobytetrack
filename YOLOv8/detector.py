from multiprocessing import Pool, Process, Semaphore, current_process, shared_memory
from queue import Empty, SimpleQueue
import numpy as np
from rknnlite.api import RKNNLite
from queue import Queue


_rknn = None
_worker_func = None


def init_rknn_worker(det_model: str, core_ids: tuple, func):
    """
    Pool initializer: runs once in each worker process.
    - Loads the RKNN model into the worker.
    - Initializes the runtime on a per-worker NPU core.
    - Stores the user inference function.
    """
    global _rknn, _worker_func

    # Determine this worker's index (1-based) and map to a core_id
    idx = current_process()._identity[0] - 1
    core_id = core_ids[idx % len(core_ids)]

    # Load and initialize RKNN
    rknn = RKNNLite()
    if rknn.load_rknn(det_model) != 0:
        raise RuntimeError(f"[Worker {idx}] Failed to load RKNN model '{det_model}'")
    core_map = {
        0: RKNNLite.NPU_CORE_0,
        1: RKNNLite.NPU_CORE_1,
        2: RKNNLite.NPU_CORE_2,
        -1: RKNNLite.NPU_CORE_0_1_2,
    }
    if rknn.init_runtime(core_mask=core_map.get(core_id, RKNNLite.NPU_CORE_0)) != 0:
        raise RuntimeError(
            f"[Worker {idx}] Failed to init RKNN runtime on core {core_id}"
        )

    _rknn = rknn
    _worker_func = func


def _pool_task(frame: np.ndarray):
    """
    The actual inference task run in each worker process.
    Expects that init_rknn_worker has set _rknn and _worker_func.
    """
    return _worker_func(_rknn, frame)


class detectExecutor:
    def __init__(
        self,
        det_model: str,
        func,
        num_workers: int = 3,
        dtype=np.uint8,
        core_ids=(0, 1, 2),
        callback=None,
        max_in_flight: int = None,  # 上限：同时在跑的任务数
        drop_when_full: bool = True,  # 队列满时是丢帧还是阻塞
    ):
        """
        Args:
            det_model: Path to .rknn model file.
            func:      Inference function signature func(rknn, frame)->res.
            num_workers: Number of parallel worker processes.
            frame_shape:  Expected shape of each input frame (for validation).
            dtype:        dtype of the frame numpy array.
            core_ids:     Tuple of core IDs to round-robin assign to workers.
            callback:     Callable(res, frame) invoked in main process.
        """

        self.dtype = dtype
        self.callback = callback
        self.drop_when_full = drop_when_full

        if max_in_flight is None:
            max_in_flight = num_workers * 2
        self.sema = Semaphore(max_in_flight)

        # Create the Pool; each worker runs init_rknn_worker once
        self.pool = Pool(
            processes=num_workers,
            initializer=init_rknn_worker,
            initargs=(det_model, core_ids, func),
        )

    def put(self, frame: np.ndarray, frame_id: int):
        """
        Submit one frame to the worker pool for inference.
        The provided callback(res, frame) will be called in the main process
        as soon as inference finishes.

        Submit one frame + its ID to the pool.
        回调会收到 (res, frame, frame_id) 三个参数。
        """

        # 尝试获取信号量：代表「一个新任务」要入列
        acquired = self.sema.acquire(block=not self.drop_when_full)

        if not acquired:
            # 丢帧策略：如果拿不到，就直接返回
            return

            # 提交给 Pool，callback 里释放信号量

        def _on_done(res, f=frame, fid=frame_id):
            try:
                if self.callback:
                    self.callback(res, fid)
            finally:
                # 一定要在 callback 后释放信号量
                self.sema.release()

        self.pool.apply_async(_pool_task, args=(frame,), callback=_on_done)

        # Submit to Pool; wrap the user callback so it receives (res, frame)
        # if self.callback:
        #     self.pool.apply_async(
        #         _pool_task,
        #         args=(frame,),
        #         callback=lambda res, fid=frame_id: self.callback(res, fid),
        #     )
        # else:
        #     # If no callback provided, just fire-and-forget
        #     self.pool.apply_async(_pool_task, args=(frame,))

    def release(self):
        """
        Gracefully shut down the Pool and wait for all tasks to complete.
        """
        self.pool.close()
        self.pool.join()
