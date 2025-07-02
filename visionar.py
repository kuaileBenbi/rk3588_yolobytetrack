import math
import logging
from pathlib import Path
import traceback
import cv2
import numpy as np

from utils import draw, CLASSES
from YOLOv8 import detectExecutor, myFunc
from tracker import BYTETracker


_CURRENT_DIR = Path(__file__).parent
DET_MODEL_PATH = _CURRENT_DIR / "rknnModel/yolov8n.rknn"

TPEs = 3

logger = logging.getLogger(__name__)

# 定义类别与位的映射
CLASS_BIT_MAPPING = {
    "boat": 0,  # Bit0
    "aeroplane": 1,  # Bit1
    "car": 2,  # Bit2
    "person": 3,
    # 可以根据需要添加更多类别
}

CLASS_BIT_MASKS = {cls: 1 << bit for cls, bit in CLASS_BIT_MAPPING.items()}


class VisionTracker:
    def __init__(self, TPEs=3, external_callback=None):
        self.external_callback = external_callback
        self.detector = detectExecutor(
            det_model=DET_MODEL_PATH,
            num_workers=TPEs,
            func=myFunc,
            callback=self._on_det_and_track,
            max_in_flight=TPEs,
            drop_when_full=True,
        )
        self.tracking = False
        self.target_id = None
        self.mask = 0xFFFFFFFF  # 默认所有类别开启
        self.TPEs = TPEs
        self.frame_cout = 0

        self.tracker = BYTETracker()

    def set_tracking_mode(self, tracking: bool):
        """切换是否开启跟踪模式"""
        self.tracking = tracking

    def set_target_mask(self, mask: int):
        """
        1-开 0-关
        设置目标掩码。每个bit代表一个类别：
        Bit0：舰船
        Bit1：飞机
        Bit2：车辆
        Bit3: 人
        当self.mask == 0xFFFFFFFF时，所有索引都被保留。

        否则，过滤条件为类别在CLASS_BIT_MAPPING中，且对应位被设置。
        """
        self.mask = mask

    def filter_mask(self, classes_name):
        if self.mask == 0xFFFFFFFF:  # 全开模式
            filtered_indices = np.arange(len(classes_name))
        else:  # 按位过滤模式

            class_names = np.array(classes_name)
            try:
                # 生成双重条件掩码
                in_mapping_mask = np.vectorize(lambda x: x in CLASS_BIT_MASKS)(
                    class_names
                )
                bit_check = np.vectorize(
                    lambda x: (self.mask & CLASS_BIT_MASKS.get(x, 0)) != 0
                )(class_names)

                # 组合条件
                mask_condition = in_mapping_mask & bit_check
                filtered_indices = np.where(mask_condition)[0]

            except Exception as e:
                logger.debug("过滤mask失败, 不过滤目标")
                filtered_indices = np.arange(len(classes_name))

        return filtered_indices

    def filter_arrays(self, arr, filtered_indices):
        """根据索引过滤数组或列表"""
        if isinstance(arr, np.ndarray):
            return arr[filtered_indices] if len(arr) > 0 else np.empty((0, 4))
        elif isinstance(arr, list):
            return [arr[i] for i in filtered_indices]
        else:
            return filtered_indices

    @staticmethod
    def deconstruct_bytetrack_outputs(online_targets):
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            online_cls.append(t.class_name)

        return online_tlwhs, online_ids, online_scores, online_cls

    def _on_det_and_track(self, det_res, frame_id):
        """
        只要检测完成，这个函数就会被执行一次：
          - 做筛选
          - 可选做跟踪
          - 最终绘图或数据发送
        """
        cur_frame, results = det_res
        # 1. 无检测时直接返回
        raw_boxes = results.get("ltrb_boxes", [])
        raw_ids = results.get("classes_id", [])
        raw_scores = results.get("scores", [])
        raw_names = [CLASSES[i] for i in raw_ids]

        if len(raw_boxes) == 0:
            self.external_callback(
                {
                    "frame": cur_frame,
                    "boxes": [],
                    "classes": [],
                },
                False,
                frame_id,
            )
            return

        # 2. 按你自己的逻辑做过滤
        keep = self.filter_mask(raw_names)
        boxes = self.filter_arrays(raw_boxes, keep)
        ids = self.filter_arrays(raw_ids, keep)
        scores = self.filter_arrays(raw_scores, keep)
        names = self.filter_arrays(raw_names, keep)

        # 3. 如果不跟踪，直接绘检测框
        if not self.tracking:
            out = draw(cur_frame, boxes, names, None)
            self.external_callback(
                {"frame": out, "boxes": boxes, "classes": names}, True, frame_id
            )
            return

        # 4. 用 ByteTrack 更新
        frm = {"ltrb_boxes": boxes, "classes_id": ids, "scores": scores}
        online_targets, ok = self.tracker.update(frm)
        if not ok:
            self.self.external_callback(
                {
                    "frame": cur_frame,
                    "boxes": [],
                    "classes": [],
                },
                False,
                frame_id,
            )
            return

        # 5. 拆解跟踪结果并绘图
        tlwhs, track_ids, track_scores, track_cls = self.deconstruct_bytetrack_outputs(
            online_targets
        )
        track_names = [CLASSES[c] for c in track_cls]
        out = draw(cur_frame, tlwhs, track_names, track_ids, tracking=True)

        # 6. 最后将结果推给下游
        self.external_callback(
            {"frame": out, "boxes": boxes, "classes": track_names}, True, frame_id
        )

    def detworking(self, img, frame_id):

        self.detector.put(img, frame_id)

    def stop_detworking(self):
        try:
            self.detector.release()

        except Exception as e:
            logger.warning(f"关闭目标检测进程失败: {e}")
            return False
        return True
