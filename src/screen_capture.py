"""
screen_capture.py — 擷取 Hollow Knight 遊戲視窗畫面

回傳 (144, 256, 1) uint8 numpy array（灰階），供 hk_env._get_screen() 使用。

設計原則：
  - 只擷取遊戲的 client area（不含標題列、邊框）
  - 使用 mss 截圖（比 PIL.ImageGrab 快約 3~5 倍）
  - 使用 OpenCV 縮放（INTER_AREA 在縮小時品質最好）
  - mss 實例在 module 層級建立一次，避免每步重複初始化的開銷
"""

import ctypes
import ctypes.wintypes

import cv2
import mss
import numpy as np

WINDOW_TITLE = "Hollow Knight"
TARGET_SIZE  = (256, 144)        # (width, height)，cv2.resize 的格式；保持 16:9 比例

# module-level mss instance，跨呼叫共用
# 【為什麼不每次建立？】
# mss.mss() 初始化需要連接 display，在 Windows 上約 5~15ms，
# 每步建立一次會把 100ms 的 step 拉長很多。
_sct = mss.mss()


def _get_client_rect(title: str) -> tuple[int, int, int, int]:
    """
    回傳遊戲視窗 client area 在螢幕上的絕對座標 (left, top, width, height)。

    【client area vs window rect 的差異】
    GetWindowRect 包含標題列和邊框，截到的畫面會有 UI chrome。
    GetClientRect 只有實際遊戲畫面，是我們要的。
    但 GetClientRect 回傳的是相對座標（左上角永遠是 0,0），
    所以需要再用 ClientToScreen 換算成螢幕絕對座標。

    找不到視窗時拋出 RuntimeError。
    """
    user32 = ctypes.windll.user32

    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        raise RuntimeError(
            f"找不到視窗：'{title}'，請確認 Hollow Knight 已開啟。"
        )

    # 取得 client area 的大小（相對座標）
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))

    # 把 client area 左上角轉成螢幕絕對座標
    origin = ctypes.wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))

    w = rect.right  - rect.left
    h = rect.bottom - rect.top
    return origin.x, origin.y, w, h


def capture_game_frame() -> tuple[np.ndarray, np.ndarray]:
    """
    擷取 Hollow Knight 視窗畫面，同時回傳灰階和彩色兩種格式。

    回傳：
        gray  : shape=(144, 256, 1), dtype=uint8  供 CNN frame stacking 使用
        color : shape=(144, 256, 3), dtype=uint8  RGB，供 Cutie 物件追蹤使用

    【為什麼一次截圖回傳兩種格式？】
    截圖本身是最耗時的操作（mss.grab），重複截圖會浪費時間。
    一次截圖後分別轉換成灰階和彩色，讓 CNN 和 Cutie 各取所需。

    【為什麼 Cutie 需要彩色？】
    Cutie 用彩色影像訓練，灰階會讓它的物件追蹤效果大幅下降。
    """
    left, top, w, h = _get_client_rect(WINDOW_TITLE)

    # mss 截圖（回傳 BGRA 格式）
    region = {"left": left, "top": top, "width": w, "height": h}
    raw = _sct.grab(region)

    # BGRA → BGR（去掉 alpha）
    frame = np.array(raw)[:, :, :3]

    # 灰階：給 CNN
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray    = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray    = gray[:, :, np.newaxis]   # (144, 256, 1)

    # 彩色：給 Cutie（BGR → RGB，縮放到同樣大小）
    color   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color   = cv2.resize(color, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # shape: (144, 256, 3)

    return gray, color


# ── 單獨執行時：顯示截圖預覽並印出耗時 ────────────────────────────────────────

if __name__ == "__main__":
    import time

    print(f"嘗試截取視窗：'{WINDOW_TITLE}'")
    try:
        # 暖機一次（第一次呼叫通常比較慢）
        capture_game_frame()

        # 計時 10 次取平均
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            gray, color = capture_game_frame()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"截圖成功！gray={gray.shape}, color={color.shape}")
        print(f"平均耗時：{avg_ms:.1f}ms（目標：< 30ms）")

        # 用 OpenCV 視窗預覽（放大 2 倍，方便肉眼確認）
        preview = cv2.resize(gray[:, :, 0], (512, 288), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("HK Frame Preview (256x144 → 512x288)", preview)
        print("按任意鍵關閉預覽視窗...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except RuntimeError as e:
        print(f"錯誤：{e}")
