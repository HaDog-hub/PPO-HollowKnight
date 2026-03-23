"""
preview_ai.py — 即時預覽 AI 看到的畫面（4 幀並排）

使用方式：
    python src/preview_ai.py

按 Q 關閉視窗。
"""

import cv2
import numpy as np
from collections import deque
from screen_capture import capture_game_frame
from hk_env import FRAME_STACK

buf = deque([np.zeros((144, 256, 1), dtype=np.uint8)] * FRAME_STACK, maxlen=FRAME_STACK)

print(f"AI 視角預覽中（{FRAME_STACK} 幀並排，左舊右新）... 按 Q 離開")
while True:
    gray, _ = capture_game_frame()
    buf.append(gray)
    row = np.concatenate([f[:, :, 0] for f in buf], axis=1)
    cv2.imshow(f"AI sees: frame 1~{FRAME_STACK} (left=oldest, right=latest)", row)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
