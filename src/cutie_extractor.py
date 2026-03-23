"""
cutie_extractor.py — Cutie 物件追蹤封裝

使用 Cutie 對每幀遊戲畫面進行物件分割追蹤，
輸出騎士和 Hornet 的二值遮罩，作為 screen obs 的額外 channel。

輸入：(H, W, 3) uint8 RGB 彩色幀
輸出：(144, 256, num_objects) uint8，每個 channel 是一個物件的二值遮罩（0 或 255）
"""

import os
import glob

import cv2
import numpy as np
import torch
from PIL import Image

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


# DAVIS 調色盤前幾個顏色，用於把 RGB 遮罩反向查找成 object ID
# Cutie GUI 存的遮罩是 palette PNG，用這些顏色區分物件
_DAVIS_PALETTE = np.array([
    [0,   0,   0  ],   # 0：背景
    [128, 0,   0  ],   # 1：騎士（深紅）
    [0,   128, 0  ],   # 2：Hornet（深綠）
    [128, 128, 0  ],   # 3：（備用）
], dtype=np.uint8)

TARGET_SIZE = (256, 144)   # (width, height)，與 screen_capture.py 一致


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def _image_to_torch(frame: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (3, H, W) float32 tensor in [0, 1]"""
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .float()
        .to(device) / 255.0
    )


def _index_to_one_hot(mask: np.ndarray, num_classes: int) -> torch.Tensor:
    """(H, W) indexed uint8 → (num_classes, H, W) float32 one-hot tensor"""
    mask_t = torch.from_numpy(mask).long()
    one_hot = torch.zeros(num_classes, *mask.shape, dtype=torch.float32)
    one_hot.scatter_(0, mask_t.unsqueeze(0), 1.0)
    return one_hot


def _load_mask_indexed(path: str) -> np.ndarray:
    """
    載入遮罩 PNG，回傳 indexed 格式 (H, W) uint8，值為 0, 1, 2...

    Cutie GUI 存的遮罩分兩種格式：
    - Palette PNG：PIL 讀出來直接是 0, 1, 2（最常見）
    - RGB PNG：使用 DAVIS 調色盤，需要反向查找 object ID
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        # 已經是 indexed（palette mode 或 grayscale）
        return arr.astype(np.uint8)

    # RGB 格式：用 DAVIS palette 反向查找
    h, w = arr.shape[:2]
    indexed = np.zeros((h, w), dtype=np.uint8)
    for obj_id, color in enumerate(_DAVIS_PALETTE):
        match = np.all(arr[:, :, :3] == color, axis=2)
        indexed[match] = obj_id
    return indexed


# ── 主類別 ────────────────────────────────────────────────────────────────────

class CutieExtractor:
    """
    Cutie 物件追蹤封裝。

    使用方式：
        extractor = CutieExtractor("labels/hornet", num_objects=2)

        # 每幀呼叫：
        masks = extractor.extract(color_frame)  # (144, 256, 2) uint8

        # 每局結束時呼叫：
        extractor.reset()
    """

    def __init__(self, label_folder: str, num_objects: int = 2):
        """
        Args:
            label_folder : 標注資料夾路徑，內含 imgs/ 和 masks/ 子資料夾
            num_objects  : 追蹤的物件數量（騎士 + Boss = 2）
        """
        self.num_objects = num_objects
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[Cutie] 載入預訓練模型...")
        cutie_model = get_default_model()
        self._processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)

        print("[Cutie] 讀入標注幀，建立永久物件記憶...")
        self._init_from_labels(label_folder)
        print("[Cutie] 初始化完成。")

    def _init_from_labels(self, label_folder: str) -> None:
        """
        讀入標注幀（imgs/）和對應遮罩（masks/），
        以 force_permanent=True 建立永久物件記憶。

        永久記憶在整個訓練過程中不會被清除，
        確保 Cutie 隨時都認得騎士和 Hornet。
        """
        img_paths = sorted(
            glob.glob(os.path.join(label_folder, "imgs", "*.jpg")) +
            glob.glob(os.path.join(label_folder, "imgs", "*.png"))
        )
        mask_paths = sorted(
            glob.glob(os.path.join(label_folder, "masks", "*.png"))
        )

        if not img_paths:
            raise FileNotFoundError(f"找不到標注圖片：{label_folder}/imgs/")
        if not mask_paths:
            raise FileNotFoundError(f"找不到遮罩：{label_folder}/masks/")
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                f"圖片數量（{len(img_paths)}）與遮罩數量（{len(mask_paths)}）不符。"
            )

        with torch.inference_mode():
            for img_path, mask_path in zip(img_paths, mask_paths):
                # 讀入彩色圖（RGB），縮放到 TARGET_SIZE
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                frame_torch = _image_to_torch(frame, self.device)

                # 讀入遮罩（indexed），縮放到 TARGET_SIZE
                # INTER_NEAREST 確保縮放後遮罩值仍為整數 0/1/2
                mask = _load_mask_indexed(mask_path)
                mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                mask_torch = _index_to_one_hot(mask, self.num_objects + 1).to(self.device)

                # 送進 Cutie 建立永久記憶
                # mask_torch[1:] 去掉背景 channel，只保留物件的 one-hot
                self._processor.step(
                    frame_torch,
                    mask_torch[1:],
                    idx_mask=False,
                    force_permanent=True,
                )

    def extract(self, color_frame: np.ndarray) -> np.ndarray:
        """
        對單幀彩色畫面進行物件追蹤，回傳各物件的二值遮罩。

        Args:
            color_frame: (H, W, 3) uint8 RGB

        Returns:
            masks: (144, 256, num_objects) uint8
                   每個 channel 對應一個物件，值為 0（無）或 255（有）
                   channel 0 = 騎士，channel 1 = Hornet
        """
        frame = cv2.resize(color_frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        frame_torch = _image_to_torch(frame, self.device)

        with torch.inference_mode():
            prediction = self._processor.step(frame_torch)

        # prediction: (num_objects+1, H, W) 機率張量
        # argmax → (H, W) indexed，值為 0（背景）, 1（騎士）, 2（Hornet）
        indexed = torch.argmax(prediction, dim=0).cpu().numpy().astype(np.uint8)

        # 拆成每個物件的二值遮罩，stacked 成 (H, W, num_objects)
        masks = np.zeros((*indexed.shape, self.num_objects), dtype=np.uint8)
        for i in range(self.num_objects):
            masks[:, :, i] = (indexed == (i + 1)) * 255

        return masks   # (144, 256, 2)

    def reset(self) -> None:
        """
        清除這局的動態記憶，保留永久標注記憶。

        每局開始前（HKEnv.reset()）呼叫，
        避免上一局的追蹤狀態污染新的一局。
        """
        self._processor.clear_non_permanent_memory()
