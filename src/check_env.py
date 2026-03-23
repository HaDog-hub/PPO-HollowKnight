"""
check_env.py — 不開遊戲，驗證 HKEnv 符合 Gym 規範

使用方式：
    python src/check_env.py

【為什麼需要這支腳本？】
SB3 的 PPO 對環境有嚴格要求（obs shape / dtype / reward 型別等）。
如果環境格式有誤，訓練時才爆掉很難 debug。
這支腳本用「假資料」代替真實遊戲連線，在不開遊戲的情況下提前抓問題。

【什麼是 Mock？】
Mock 是測試中的「替身」技術：把真實的元件（TCP 連線、鍵盤輸入）
換成一個假的物件，讓它回傳我們預先設定的假資料。
這樣就能測試「環境的邏輯」，而不需要真的連接遊戲。
"""

from unittest.mock import MagicMock, patch

import numpy as np
from stable_baselines3.common.env_checker import check_env

from hk_env import HKEnv


# ── 假資料：模擬 Mod 推送的遊戲狀態 ──────────────────────────────────────────
#
# 這些字典的結構要和真實 hk_client.py 回傳的完全一致，
# 因為 HKEnv._calc_reward() 和 _make_obs() 都依賴這個格式。

STATE_NORMAL = {
    "type": "event",
    "timestamp": 1700000000000,
    "scene": "GG_Hornet_1",
    "in_boss_room": True,
    "player": {"hp": 5, "max_hp": 5, "hp_pct": 1.0},
    "boss":   {"name": "Hornet Boss 1", "hp": 700, "max_hp": 700, "hp_pct": 1.0},
}

STATE_PLAYER_HIT = {
    **STATE_NORMAL,
    "player": {"hp": 4, "max_hp": 5, "hp_pct": 0.8},
}

STATE_BOSS_HIT = {
    **STATE_NORMAL,
    "boss": {"name": "Hornet Boss 1", "hp": 600, "max_hp": 700, "hp_pct": 0.857},
}

STATE_PLAYER_DEAD = {
    **STATE_NORMAL,
    "player": {"hp": 0, "max_hp": 5, "hp_pct": 0.0},
}

STATE_BOSS_DEAD = {
    **STATE_NORMAL,
    "boss": {"name": "Hornet Boss 1", "hp": 0, "max_hp": 700, "hp_pct": 0.0},
}


def make_mock_env(initial_state=STATE_NORMAL) -> HKEnv:
    """
    建立一個使用假資料的 HKEnv。

    用 patch 替換兩個真實元件：
      1. HKBridgeClient — 不做 TCP 連線，get_latest_state() 回傳假資料
      2. pynput.Controller — 不做真實鍵盤輸入（避免測試時亂按）
    """
    # MagicMock() 會自動幫所有方法建立假實作
    mock_client = MagicMock()
    mock_client.get_latest_state.return_value = initial_state

    # patch 的作用：在 hk_env 模組裡，把 HKBridgeClient 換成回傳 mock_client 的版本
    # patch 的目標格式是「模組路徑.類別名稱」
    # mock_screen：回傳正確 shape 的假畫面（全黑），避免測試時真的去截圖
    with patch("hk_env.HKBridgeClient", return_value=mock_client), \
         patch("hk_action.Controller"), \
         patch("hk_env.CutieExtractor"):
        env = HKEnv()
        env._client = mock_client
        # 直接替換截圖方法：回傳固定全黑幀，不需要遊戲視窗
        # shape = (144, 256, FRAME_STACK + NUM_OBJECTS) = (144, 256, 10)
        env._get_stacked_screen = lambda: np.zeros((144, 256, 10), dtype=np.uint8)
    return env


# ── 測試 1：SB3 官方環境檢查 ──────────────────────────────────────────────────

def test_sb3_check():
    """
    執行 SB3 內建的 check_env()。

    這個函式會自動驗證：
      - reset() 回傳的 obs 形狀符合 observation_space
      - step() 回傳的五個值型別正確
      - obs 的值在 observation_space 宣告的範圍內
      - reward 是純量 float
      - terminated / truncated 是 bool
      ... 等 20 幾項規範
    """
    print("=" * 50)
    print("測試 1：SB3 check_env()")
    print("=" * 50)

    env = make_mock_env()
    check_env(env, warn=True)   # warn=True：把警告也印出來
    print("[OK] check_env 通過\n")


# ── 測試 2：手動驗證 obs 的 shape 和 dtype ────────────────────────────────────

def test_obs_format():
    """
    【為什麼要手動測 shape 和 dtype？】
    check_env 已經做了，但手動印出來讓你眼見為憑，
    也方便之後加入真實畫面時確認格式有沒有改變。
    """
    print("=" * 50)
    print("測試 2：Observation 格式")
    print("=" * 50)

    env = make_mock_env()
    obs, info = env.reset()

    # screen
    screen = obs["screen"]
    print(f"screen shape : {screen.shape}   (預期: (144, 256, 10))")
    print(f"screen dtype : {screen.dtype}   (預期: uint8)")
    assert screen.shape == (144, 256, 10), f"shape 錯誤: {screen.shape}"
    assert screen.dtype == np.uint8,    f"dtype 錯誤: {screen.dtype}"

    # stats
    stats = obs["stats"]
    print(f"stats  shape : {stats.shape}    (預期: (2,))")
    print(f"stats  dtype : {stats.dtype}  (預期: float32)")
    print(f"stats  值    : {stats}         (預期: [1.0, 1.0] 滿血初始)")
    assert stats.shape == (2,),          f"shape 錯誤: {stats.shape}"
    assert stats.dtype == np.float32,    f"dtype 錯誤: {stats.dtype}"
    assert np.all(stats >= 0) and np.all(stats <= 1), "stats 超出 [0, 1] 範圍"

    print("[OK] Observation 格式正確\n")


# ── 測試 3：Reward 計算邏輯 ────────────────────────────────────────────────────

def test_reward_logic():
    """
    手動測試各種情境下的 reward，確認計算邏輯正確。

    【為什麼要逐一測試情境？】
    Reward 的 bug 很隱蔽：數值算錯不會讓程式崩潰，
    只會讓 AI 學到奇怪的行為，而且很難追查原因。
    提前驗證每個情境的 reward 符合預期，可以省掉大量訓練時間。
    """
    print("=" * 50)
    print("測試 3：Reward 計算邏輯")
    print("=" * 50)

    env = make_mock_env(STATE_NORMAL)
    env.reset()

    def one_step(state, label):
        env._client.get_latest_state.return_value = state
        _, reward, terminated, truncated, _ = env.step(0)  # 動作 0 = IDLE
        done = terminated or truncated
        print(f"  {label:20s}  reward={reward:+7.3f}  done={done}  "
              f"term={terminated}  trunc={truncated}")
        return reward, terminated, truncated

    # 什麼都沒發生（IDLE 一步）
    r, term, trunc = one_step(STATE_NORMAL, "IDLE（無變化）")
    assert abs(r - (-0.005)) < 0.001, f"IDLE reward 應約為 -0.005，got {r}"
    assert not term and not trunc

    # 重新 reset 讓 prev HP 歸位
    env._client.get_latest_state.return_value = STATE_NORMAL
    env.reset()

    # 玩家被打
    r, term, trunc = one_step(STATE_PLAYER_HIT, "玩家被打（-0.2 HP）")
    assert r < -0.01, "被打應該得到負 reward"
    assert not term

    env._client.get_latest_state.return_value = STATE_NORMAL
    env.reset()

    # 打到 Boss
    r, term, trunc = one_step(STATE_BOSS_HIT, "打到 Boss")
    assert r > 0, "打到 Boss 應該得到正 reward"
    assert not term

    env._client.get_latest_state.return_value = STATE_NORMAL
    env.reset()

    # 玩家死亡
    r, term, trunc = one_step(STATE_PLAYER_DEAD, "玩家死亡")
    assert term, "玩家死亡應該 terminated=True"
    assert r < -10, f"死亡應有大負分，got {r}"

    env._client.get_latest_state.return_value = STATE_NORMAL
    env.reset()

    # Boss 死亡（贏了！）
    # 要先把步數設超過 MIN_EPISODE_STEPS，否則會被最短局時保護擋下來
    from hk_env import MIN_EPISODE_STEPS
    env._step_count = MIN_EPISODE_STEPS
    r, term, trunc = one_step(STATE_BOSS_DEAD, "Boss 死亡（贏）")
    assert term, "Boss 死亡應該 terminated=True"
    assert r > 10, f"獲勝應有大正分，got {r}"

    print("[OK] Reward 邏輯全部正確\n")


# ── 測試 4：動作空間 ──────────────────────────────────────────────────────────

def test_action_space():
    print("=" * 50)
    print("測試 4：動作空間")
    print("=" * 50)

    env = make_mock_env()
    print(f"action_space      : {env.action_space}")
    print(f"動作數量          : {env.action_space.n}  (預期: 15)")
    assert env.action_space.n == 15, f"動作數量錯誤: {env.action_space.n}"

    # 確認每個 action_id 都能執行不會炸
    env.reset()
    for action_id in range(env.action_space.n):
        env._client.get_latest_state.return_value = STATE_NORMAL
        env.step(action_id)

    print(f"[OK] 所有 {env.action_space.n} 個動作執行正常\n")


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sb3_check()
    test_obs_format()
    test_reward_logic()
    test_action_space()

    print("=" * 50)
    print("[OK] 全部測試通過！HKEnv 符合 Gym 規範，可以開始訓練。")
    print("=" * 50)
