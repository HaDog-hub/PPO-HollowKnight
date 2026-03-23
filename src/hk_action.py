"""
hk_action.py — Hollow Knight AI 動作執行模組

設計原則：
  - 純離散動作空間，每個 action_id 對應一組按鍵 + 各自的按住時長
  - execute() 是同步阻塞呼叫，一個 step = 100ms
  - 使用 pynput 模擬鍵盤輸入
  - 若遊戲無反應，可將 _press / _release 換成 ctypes scan code 版本
"""

import ctypes
import time
from dataclasses import dataclass, field
from typing import Union

from pynput.keyboard import Controller, Key


# ── 遊戲視窗焦點偵測 ─────────────────────────────────────────────────────────

def _get_foreground_title() -> str:
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def is_game_focused() -> bool:
    return "Hollow Knight" in _get_foreground_title()


def wait_for_game_focus() -> None:
    """
    阻塞直到 Hollow Knight 視窗重新取得焦點。

    訓練期間若畫面失焦，呼叫此函式可暫停動作輸入，
    防止鍵盤指令被送到其他視窗（例如編輯器、瀏覽器）。
    每 0.5 秒偵測一次，重新取得焦點後立即返回。
    """
    if is_game_focused():
        return
    print("[HKAction] 遊戲視窗失焦，暫停輸入，等待重新對焦...")
    while not is_game_focused():
        time.sleep(0.5)
    print("[HKAction] 遊戲視窗重新對焦，繼續訓練。")


# ── 鍵位對應表 ──────────────────────────────────────────────────────────────
#
# 【為什麼要有這張表？】
# 這是一層「抽象」。我們在動作定義裡用 "jump"、"attack" 這種語意名稱，
# 而不是直接寫 "z"、"x"。好處是：如果玩家改了遊戲鍵位，
# 只需要改這裡一個地方，動作表完全不需要動。
#
# 【為什麼特殊鍵（left/right）要用 Key enum？】
# pynput 對方向鍵等特殊鍵用 Key.left、Key.right 這種 enum 值，
# 一般字元鍵（z、x）直接用字串 "z"、"x" 就好。

_KEY_MAP: dict[str, Union[Key, str]] = {
    "left":   Key.left,
    "right":  Key.right,
    "up":     Key.up,
    "down":   Key.down,
    "jump":   "z",      # Hollow Knight 預設跳躍鍵
    "attack": "x",      # 預設攻擊鍵
    "dash":   "c",      # 預設衝刺鍵
}

# release_all() 會用到：遊戲中所有可能被按住的鍵
_ALL_GAME_KEYS = list(_KEY_MAP.keys())


# ── 動作定義的資料結構 ───────────────────────────────────────────────────────
#
# 【為什麼用 dataclass？】
# dataclass 是 Python 的內建語法糖，幫你自動生成 __init__、__repr__ 等方法。
# 比起寫 tuple 或 dict，dataclass 讓每個欄位有名字，讀起來一目了然，
# 也方便之後加新欄位（例如：只有在空中才能執行的動作）。

@dataclass
class KeyPress:
    """單一按鍵的設定：按哪個鍵、按住幾毫秒"""
    key: str      # _KEY_MAP 中的語意名稱，例如 "jump"
    hold_ms: int  # 按住的時間（毫秒）


@dataclass
class ActionDef:
    """一個離散動作的完整定義"""
    name: str
    keys: list[KeyPress] = field(default_factory=list)
    step_ms: int = 75  # 這個 step 的總時長（毫秒）


# ── 動作表 ──────────────────────────────────────────────────────────────────
#
# 【核心設計：為什麼每個按鍵有自己的 hold_ms？】
# 以 LEFT_JUMP_TAP 為例，我們希望：
#   - 左鍵：按住整個 100ms（讓角色持續往左移動）
#   - 跳躍鍵：只按 25ms（小跳，不要跳太高）
# 如果兩個鍵都按 100ms，就只有 JUMP_FULL 等級的大跳。
# 把按住時長分開設定，AI 才能真正控制跳躍高度。
#
# 【IDLE 為什麼要存在？】
# AI 有時候需要「等待觀察」，不做任何輸入。
# 如果沒有 IDLE，AI 每個 step 都必須按某個鍵，會導致過度操作。

ACTIONS: dict[int, ActionDef] = {
    #  id  名稱                按鍵（key, hold_ms）
    0:  ActionDef("IDLE"),
    1:  ActionDef("LEFT",            [KeyPress("left",   100)]),
    2:  ActionDef("RIGHT",           [KeyPress("right",  100)]),
    3:  ActionDef("JUMP_TAP",        [KeyPress("jump",    25)]),
    4:  ActionDef("JUMP_FULL",       [KeyPress("jump",   100)]),
    5:  ActionDef("ATTACK",          [KeyPress("attack", 100)]),
    6:  ActionDef("DASH",            [KeyPress("dash",   100)]),
    7:  ActionDef("LEFT_JUMP_TAP",   [KeyPress("left",   100), KeyPress("jump",    25)]),
    8:  ActionDef("LEFT_JUMP_FULL",  [KeyPress("left",   100), KeyPress("jump",   100)]),
    9:  ActionDef("RIGHT_JUMP_TAP",  [KeyPress("right",  100), KeyPress("jump",    25)]),
    10: ActionDef("RIGHT_JUMP_FULL", [KeyPress("right",  100), KeyPress("jump",   100)]),
    11: ActionDef("LEFT_ATTACK",     [KeyPress("left",   100), KeyPress("attack", 100)]),
    12: ActionDef("RIGHT_ATTACK",    [KeyPress("right",  100), KeyPress("attack", 100)]),
    13: ActionDef("UP_ATTACK",       [KeyPress("up",     100), KeyPress("attack", 100)]),
    # 下劈：遊戲層面只有空中才能觸發 Dive Attack；地面按下去會變成普通攻擊。
    # 不在這裡強制判斷（沒有 is_airborne 資訊），讓 AI 透過 reward 自行學習何時該用。
    14: ActionDef("DOWN_ATTACK",     [KeyPress("down",   100), KeyPress("attack", 100)]),
}

ACTION_SPACE_SIZE = len(ACTIONS)  # 15，Gym env 會用到


# ── 執行器 ──────────────────────────────────────────────────────────────────

class HKActionExecutor:
    """
    將 action_id 轉換成實際鍵盤輸入。

    使用方式：
        executor = HKActionExecutor()
        executor.execute(7)   # LEFT_JUMP_TAP
        executor.release_all()  # episode 結束 / 緊急停止時呼叫
    """

    def __init__(self):
        # 【為什麼在 __init__ 呼叫 release_all？】
        # 程式啟動或重新建立 executor 時，確保遊戲中沒有任何鍵被「卡住」。
        # 如果上次訓練崩潰，可能有鍵沒有正確 release，角色會一直往某個方向跑。
        self._kb = Controller()
        self.release_all()

    def execute(self, action_id: int) -> None:
        """
        執行一個動作，阻塞直到 step_ms 結束後才返回。

        【為什麼要阻塞（synchronous）？】
        RL 的訓練迴圈是：執行動作 → 等待 → 讀取狀態 → 計算 reward → 執行下一個動作。
        這個流程天生是循序的，不需要非同步。阻塞設計讓流程更簡單，
        不需要鎖、事件、callback 等複雜機制。

        失焦時會先呼叫 release_all() 放開所有按鍵，再阻塞等待重新對焦，
        防止鍵盤輸入被送到其他視窗。
        """
        if not is_game_focused():
            self.release_all()
            wait_for_game_focus()

        if action_id not in ACTIONS:
            raise ValueError(f"未知的 action_id: {action_id}")

        action = ACTIONS[action_id]

        # IDLE：什麼都不按，只是等待
        if not action.keys:
            time.sleep(action.step_ms / 1000)
            return

        # 同時按下所有鍵
        # 【為什麼要「同時」按？】
        # LEFT + JUMP 如果有時間差，角色可能先水平移動一步再跳，
        # 而不是「斜向跳躍」。同時按才符合玩家實際操作的手感。
        for kp in action.keys:
            self._kb.press(_KEY_MAP[kp.key])

        # 依 hold_ms 由短到長排序，依序在正確時間點放開
        #
        # 【排序放開的邏輯說明】
        # 以 LEFT_JUMP_TAP（left=100ms, jump=25ms）為例：
        #   t=0ms:  同時按下 left 和 jump
        #   t=25ms: 放開 jump（小跳結束）
        #   t=100ms: 放開 left（移動結束）
        # 如果不排序，順序錯誤會導致鍵被卡住或提早放開。
        sorted_keys = sorted(action.keys, key=lambda kp: kp.hold_ms)

        elapsed_ms = 0
        for kp in sorted_keys:
            wait = kp.hold_ms - elapsed_ms
            if wait > 0:
                time.sleep(wait / 1000)
                elapsed_ms = kp.hold_ms
            self._kb.release(_KEY_MAP[kp.key])

        # 補足剩餘的 step 時間
        # 【為什麼要補足？】
        # 確保每個 step 固定是 100ms，讓 RL 的時間基準一致。
        # 否則 JUMP_TAP（25ms 按鍵）的 step 只有 25ms，
        # 比 LEFT（100ms）短了 4 倍，reward 的時間密度就不一樣了。
        remaining = action.step_ms - elapsed_ms
        if remaining > 0:
            time.sleep(remaining / 1000)

    def release_all(self) -> None:
        """
        強制放開所有遊戲按鍵。

        呼叫時機：
          - episode 結束（角色死亡 / Boss 擊敗）
          - 訓練被 Ctrl+C 中斷
          - 任何例外發生時
        """
        for key_name in _ALL_GAME_KEYS:
            try:
                self._kb.release(_KEY_MAP[key_name])
            except Exception:
                pass  # 沒按住的鍵 release 會丟例外，直接忽略即可

    @staticmethod
    def action_name(action_id: int) -> str:
        """取得動作名稱，方便 logging 用"""
        return ACTIONS[action_id].name if action_id in ACTIONS else "UNKNOWN"
