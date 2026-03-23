"""
hk_env.py — Hollow Knight Boss 戰 Gym 環境

將遊戲封裝成標準的 gymnasium.Env 介面，供 RL 演算法（PPO 等）使用。

目前目標：GG_Hornet_1（尋神者模式 Hornet Protector）

Observation：
    screen : (144, 256, 4) uint8  最近 4 幀疊加（frame stacking），灰階
    stats  : (2,) float32         [player_hp_pct, boss_hp_pct]

Action：
    Discrete(15)  ← 詳見 hk_action.py

Reward：
    每步 dense  : Boss 傷害獎勵 + 玩家受傷懲罰 + 時間懲罰
    終局 sparse : 勝利 / 死亡 加減大分
"""

import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hk_client import HKBridgeClient
from hk_action import HKActionExecutor, ACTION_SPACE_SIZE
from screen_capture import capture_game_frame
from cutie_extractor import CutieExtractor


# ── Reward 權重（集中放在這裡，之後要調只改這塊）──────────────────────────────
#
# 【為什麼要集中管理？】
# Reward 的調整是訓練過程中最頻繁的事。如果數字散落在 code 各處，
# 改起來容易漏掉，也很難一眼看清楚「這個環境的激勵結構是什麼」。

REWARD_BOSS_DMG   =  5.0   # Boss HP% 每下降 1.0 給的獎勵（一刀 ≈ +0.28）
REWARD_PLAYER_DMG =  -3.0  # 玩家 HP% 每下降 1.0 的懲罰（一格 ≈ -0.67）
                            # 從 -5 降到 -3，讓攻擊與受傷的比例從 4:1 → 2.4:1，減少對探索的抑制
REWARD_TIME_STEP  = -0.01  # 每步的時間懲罰（鼓勵積極輸出，不要龜縮）
REWARD_SURVIVAL   =  0.005 # 每步依當前玩家 HP% 給的存活獎勵（滿血 +0.005/步）
                            # 與時間懲罰搭配：血量越高淨懲罰越小，鼓勵閃避
REWARD_WIN        = +20.0  # 擊敗 Boss 的終局獎勵
REWARD_LOSE       = -20.0  # 玩家死亡的終局懲罰
REWARD_DISCONNECT = -20.0  # 連線中斷（視同失敗處理）

# Boss HP 里程碑獎勵
# 【為什麼需要里程碑？】
# AI 從來沒有贏過，REWARD_WIN 的訊號等於不存在。
# 加入里程碑讓 AI 能感受到「打到 50% 比打到 75% 更好」，
# 提供明確的進度梯度，避免 AI 永遠困在局部最優。
# 每個門檻每局只觸發一次（由 _milestones_hit 追蹤）。
REWARD_MILESTONES = {
    0.75: +3.0,
    0.65: +3.0,
    0.55: +4.0,
    0.45: +5.0,   # AI 目前平台區，加密訊號引導突破
    0.35: +6.0,
    0.25: +8.0,
    0.15: +10.0,
    0.05: +12.0,
}

# 最短局時保護
# 【為什麼需要這個？】
# 某些 Boss（如假騎士）有多個 HealthManager，Mod 有時會在入場動畫期間
# 短暫偵測到一個會快速歸零的臨時物件，導致誤判為 WIN。
# 加上這個下限：局時少於 MIN_EPISODE_STEPS 步時，
# 把「WIN」降級為 truncated（異常中斷），不給 WIN reward，
# 並印出警告讓訓練者知道有異常偵測。
MIN_EPISODE_STEPS = 30   # 3 秒（30 × 100ms），低於此視為誤判

# Boss HP 穩定等待時間（秒）
# reset() 等待 Boss 房時，要求 Boss HP 連續穩定這麼久才開始
# 避免在入場動畫期間（HP 還在跳動）就開始訓練
BOSS_STABLE_SECS  = 1.0

# 卡關超時（步數）
# 若連續這麼多步都沒有任何傷害交換（boss HP 與 player HP 均無變化），
# 視為 Boss 卡在無敵/凍結狀態，強制 truncate 這一局重置。
# 300 步 × 100ms = 30 秒；Hornet stagger 正常約 5~10 秒，30 秒絕對是卡死。
STUCK_TIMEOUT_STEPS = 300

# 單局最大步數（最後防線）
# 即使 stuck detection 與其他終止條件都沒觸發，這裡也會強制結束這一局。
# 3000 步 × 100ms = 5 分鐘；Hornet 一場正常打鬥不應超過此限。
MAX_EPISODE_STEPS = 3000

# Frame Stacking 幀數
# 連續疊 8 幀讓 CNN 看到移動方向與速度
# 時間窗口 = FRAME_STACK × 100ms = 800ms
FRAME_STACK = 8

# Cutie 追蹤物件數（騎士 + Hornet）
# screen obs 的最後 NUM_OBJECTS 個 channel 是 Cutie 輸出的二值遮罩
NUM_OBJECTS = 2

# 標注資料夾路徑（相對於 train.py 執行位置，或絕對路徑）
CUTIE_LABEL_FOLDER = "labels/hornet"


class HKEnv(gym.Env):
    """
    Hollow Knight Boss 戰 Gym 環境。

    使用方式：
        env = HKEnv()
        obs, info = env.reset()        # 等待玩家手動進入 Boss 房
        obs, reward, term, trunc, info = env.step(action)
        env.close()
    """

    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 11000):
        super().__init__()

        # ── 動作空間 ──────────────────────────────────────────────────────────
        #
        # 【為什麼用 spaces.Discrete？】
        # Discrete(n) 代表 AI 每步只能選一個整數（0 ~ n-1），
        # 對應 hk_action.py 裡的 action_id。
        # PPO、DQN 等演算法看到 Discrete space 就知道要輸出一個 softmax 分類。
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # ── 觀測空間 ──────────────────────────────────────────────────────────
        #
        # 【為什麼用 spaces.Dict？】
        # screen 是影像（整數像素），stats 是數值百分比（浮點數）。
        # 兩種資料的型別、範圍、意義完全不同，放在同一個 Box 裡會很奇怪。
        # Dict space 讓兩者各自有明確的定義，SB3 的 MultiInputPolicy
        # 會自動對 screen 用 CNN、對 stats 用 MLP，分開處理再合併。
        #
        # 【screen shape 說明】
        # (height=144, width=256, channels=FRAME_STACK + NUM_OBJECTS = 10)
        # channel 0~7：最近 8 幀灰階畫面（frame stacking），讓 CNN 感知動態
        # channel 8  ：騎士分割遮罩（Cutie 輸出，0 或 255）
        # channel 9  ：Hornet 分割遮罩（Cutie 輸出，0 或 255）
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(
                low=0, high=255,
                shape=(144, 256, FRAME_STACK + NUM_OBJECTS),
                dtype=np.uint8,
            ),
            "stats": spaces.Box(
                low=0.0, high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        })

        # ── 內部元件 ──────────────────────────────────────────────────────────
        self._client   = HKBridgeClient(host, port)
        self._executor = HKActionExecutor()
        self._cutie    = CutieExtractor(CUTIE_LABEL_FOLDER, num_objects=NUM_OBJECTS)

        # Frame buffer：固定長度 deque，滿了自動丟掉最舊的幀
        # 初始填滿全黑（代表「還沒看到任何畫面」）
        self._frame_buffer: deque = deque(
            [self._blank_frame() for _ in range(FRAME_STACK)],
            maxlen=FRAME_STACK,
        )

        # 上一步的 HP 百分比，用來計算這一步的「變化量」
        self._prev_player_hp_pct: float = 1.0
        self._prev_boss_hp_pct:   float = 1.0
        self._step_count:   int = 0   # 這局已執行的步數（用於最短局時保護）
        self._episode_count: int = 0  # 累計局數（第一局需要提示玩家走到大廳）
        self._milestones_hit: set = set()  # 這局已觸發的里程碑門檻（避免重複給獎勵）
        self._steps_no_change: int = 0    # 連續無傷害交換的步數（偵測卡關用）

    # ── 核心方法 ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        重置環境，準備開始新的一局。

        【目前的限制】
        Phase 2 才有自動重置（goto_scene 指令）。
        現在需要玩家手動在遊戲裡進入 GG_Hornet_1，
        這個方法會等待直到偵測到有效的 Boss 房狀態。
        """
        super().reset(seed=seed)

        # 放開所有按鍵，避免上一局的輸入殘留
        self._executor.release_all()

        # 建立連線（如果還沒連）
        # 【為什麼在 reset 裡連線而不是 __init__？】
        # __init__ 只是「準備好環境」，不代表遊戲已經開著。
        # reset() 才是「我要開始一局了」，這時才需要連線。
        try:
            self._client.connect()
        except OSError:
            pass  # Mod 未啟動時 connect() 會丟 OSError，忽略後等 _wait_for_boss_room 逾時

        # 等待 Boss 房就緒
        # 第一局：Mod 不知道要切場景，提示玩家先到大廳，之後 Mod 自動處理
        # 第二局起：玩家死亡 → 遊戲回到 GG_Atrium → Mod 自動載入 GG_Hornet_1
        if self._episode_count == 0:
            print("[HKEnv] 請在遊戲中前往 Godhome 大廳（GG_Atrium），Mod 將自動切換場景...")
        self._episode_count += 1

        # 逾時不 crash，自動重試，直到 Boss 房就緒為止
        # 【為什麼需要重試而不是直接拋錯？】
        # 打贏 Boss 後勝利動畫 / 過場可能比死亡路徑更長，
        # Mod 回到 Atrium 的時機不確定。若只等一次 120 秒就 crash，
        # 無人值守訓練（過夜）會因為第一次勝利而整個中止。
        while True:
            try:
                state = self._wait_for_boss_room()
                break
            except TimeoutError as e:
                print(f"[HKEnv] {e}  →  重試中，請確認遊戲與 Mod 狀態...")
                try:
                    self._client.connect()
                except OSError:
                    pass

        # 初始化 HP 追蹤
        self._prev_player_hp_pct = state["player"]["hp_pct"]
        self._prev_boss_hp_pct   = state["boss"]["hp_pct"]
        self._step_count         = 0
        self._milestones_hit     = set()
        self._steps_no_change    = 0

        # Frame buffer 清空：新的一局，舊的畫面不應該影響初始觀測
        for i in range(FRAME_STACK):
            self._frame_buffer[i] = self._blank_frame()

        # Cutie 重置：清除上一局的動態追蹤記憶，保留永久標注記憶
        self._cutie.reset()

        return self._make_obs(state), {}

    def step(self, action: int):
        """
        執行一個動作，回傳五個值（Gymnasium 新版 API）。

        回傳：
            obs        : 新的觀測
            reward     : 這個 step 的得分
            terminated : 正常結束（Boss 死 or 玩家死）
            truncated  : 異常中斷（離開 Boss 房 / 連線斷）
            info       : 額外資訊（方便 debug，不影響訓練）

        【terminated vs truncated 的差異】
        這是 Gymnasium 在 v0.26 引入的重要區分：
        - terminated=True：遊戲有個明確的結局（贏/輸），這局「自然結束」
        - truncated=True ：不是因為遊戲結局而停止，而是外部原因中斷
        RL 演算法對這兩種結束的處理方式不同，
        terminated 會讓 value function 學到「這是終點」，
        truncated 則會用 bootstrap 估計「如果繼續會怎樣」。
        """
        # 1. 執行動作（阻塞 100ms）
        self._executor.execute(action)

        # 2. 讀取新狀態
        state = self._client.get_latest_state()

        # 連線中斷時，直接結束這一局
        if state is None:
            return self._empty_obs(), REWARD_DISCONNECT, False, True, {"reason": "disconnect"}

        # 3. 計算 reward 並判斷結束條件
        self._step_count += 1
        reward, terminated, truncated = self._calc_reward(state)

        # 4. 更新 HP 追蹤（下一步計算 delta 會用到）
        self._prev_player_hp_pct = state["player"]["hp_pct"]
        self._prev_boss_hp_pct   = state["boss"]["hp_pct"]

        info = {
            "player_hp":   state["player"]["hp"],
            "boss_hp":     state["boss"]["hp"],
            "boss_hp_pct": state["boss"]["hp_pct"],
            "scene":       state["scene"],
        }

        return self._make_obs(state), reward, terminated, truncated, info

    def close(self):
        """釋放所有資源"""
        self._executor.release_all()
        self._client.disconnect()

    # ── Reward 計算 ──────────────────────────────────────────────────────────

    def _calc_reward(self, state: dict) -> tuple[float, bool, bool]:
        """
        計算這個 step 的 reward，以及是否結束。

        回傳 (reward, terminated, truncated)
        """
        curr_player_hp_pct = state["player"]["hp_pct"]
        curr_boss_hp_pct   = state["boss"]["hp_pct"]

        # 計算「這一步的變化量」
        # 正值 = 好事發生（boss 掉血 / 玩家... 沒有正的啦）
        # 注意：hp_pct 是 0~1，減少代表受傷，所以 prev - curr = 傷害量
        boss_dmg   = self._prev_boss_hp_pct   - curr_boss_hp_pct
        player_dmg = self._prev_player_hp_pct - curr_player_hp_pct

        # 卡關偵測：連續無傷害交換計數
        if boss_dmg == 0 and player_dmg == 0:
            self._steps_no_change += 1
        else:
            self._steps_no_change = 0

        # Dense reward（每步都算）
        reward  = boss_dmg            * REWARD_BOSS_DMG    # 打到 Boss 加分
        reward += player_dmg          * REWARD_PLAYER_DMG  # 被打扣分
        reward += REWARD_TIME_STEP                          # 時間懲罰
        reward += curr_player_hp_pct  * REWARD_SURVIVAL    # 存活獎勵

        # 里程碑獎勵（每個門檻每局只觸發一次）
        for threshold, bonus in REWARD_MILESTONES.items():
            if threshold not in self._milestones_hit and curr_boss_hp_pct < threshold:
                reward += bonus
                self._milestones_hit.add(threshold)

        terminated = False
        truncated  = False

        # 終局判斷
        player_dead = state["player"]["hp"] <= 0
        boss_dead   = state["boss"]["hp"]   <= 0
        left_room   = not state.get("in_boss_room", True)

        if boss_dead:
            # 【為什麼 boss_dead 優先檢查？】
            # 假設玩家最後一擊同時讓 boss 死、自己也死（不太可能但要考慮），
            # 優先算「贏」比較合理。

            if self._step_count < MIN_EPISODE_STEPS:
                # 【最短局時保護】
                # 太快結束代表偵測到的是入場動畫中會消失的臨時物件，
                # 不是真正的 Boss 死亡。降級為 truncated，不給 WIN reward。
                print(
                    f"[HKEnv] 警告：第 {self._step_count} 步就偵測到 boss_hp=0，"
                    f"低於最低門檻 {MIN_EPISODE_STEPS} 步，視為誤判（truncated）。"
                )
                truncated = True
            else:
                reward    += REWARD_WIN
                terminated = True
        elif player_dead:
            reward    += REWARD_LOSE
            terminated = True
        elif left_room:
            # 離開 Boss 房但沒人死，視為異常中斷（例如卡關、手動退出）
            # 不加額外懲罰，但結束這一局
            truncated = True
        elif self._steps_no_change >= STUCK_TIMEOUT_STEPS:
            # Boss 卡在無敵/凍結狀態：連續 STUCK_TIMEOUT_STEPS 步沒有任何傷害交換
            # 強制結束這一局，讓 Mod 自動重置場景
            print(
                f"[HKEnv] 警告：連續 {self._steps_no_change} 步（"
                f"{self._steps_no_change * 0.1:.0f}s）無傷害交換，"
                f"Boss HP={curr_boss_hp_pct*100:.1f}%，判定為卡關，強制 truncate。"
            )
            truncated = True
        elif self._step_count >= MAX_EPISODE_STEPS:
            # 單局最大步數保護：無論如何，超過 5 分鐘就強制結束
            print(
                f"[HKEnv] 警告：第 {self._step_count} 步，達到最大局時上限 "
                f"（{MAX_EPISODE_STEPS} 步），強制 truncate。"
            )
            truncated = True

        return reward, terminated, truncated

    # ── 觀測建構 ─────────────────────────────────────────────────────────────

    def _make_obs(self, state: dict) -> dict:
        """將遊戲狀態轉換成 observation dict"""
        return {
            "screen": self._get_stacked_screen(),
            "stats": np.array([
                state["player"]["hp_pct"],
                state["boss"]["hp_pct"],
            ], dtype=np.float32),
        }

    def _get_stacked_screen(self) -> np.ndarray:
        """
        截一幀新畫面加入 buffer，執行 Cutie 物件追蹤，回傳完整觀測。

        截圖失敗時插入全黑幀，Cutie 失敗時插入全零遮罩，訓練不中斷。
        回傳 shape：(144, 256, FRAME_STACK + NUM_OBJECTS)
            channel 0~FRAME_STACK-1 : 灰階幀疊加
            channel FRAME_STACK+    : Cutie 二值遮罩（騎士、Hornet）
        """
        try:
            gray, color = capture_game_frame()   # gray:(144,256,1), color:(144,256,3)
        except Exception as e:
            print(f"[HKEnv] 截圖失敗，插入空白幀：{e}")
            gray  = self._blank_frame()
            color = np.zeros((144, 256, 3), dtype=np.uint8)

        self._frame_buffer.append(gray)

        # Cutie 物件追蹤
        try:
            masks = self._cutie.extract(color)   # (144, 256, NUM_OBJECTS)
        except Exception as e:
            print(f"[HKEnv] Cutie 追蹤失敗，插入零遮罩：{e}")
            masks = np.zeros((144, 256, NUM_OBJECTS), dtype=np.uint8)

        # 沿 channel 軸疊合：(144,256,8) + (144,256,2) → (144,256,10)
        stacked = np.concatenate(list(self._frame_buffer), axis=2)
        return np.concatenate([stacked, masks], axis=2)

    @staticmethod
    def _blank_frame() -> np.ndarray:
        """回傳單張全黑幀 (144, 256, 1)，用於 buffer 初始化和截圖失敗的 fallback"""
        return np.zeros((144, 256, 1), dtype=np.uint8)

    def _empty_obs(self) -> dict:
        """連線中斷時回傳的空觀測（全零）"""
        return {
            "screen": np.zeros((144, 256, FRAME_STACK + NUM_OBJECTS), dtype=np.uint8),
            "stats":  np.array([0.0, 0.0], dtype=np.float32),
        }

    def _wait_for_boss_room(self, timeout: float = 120.0) -> dict:
        """
        阻塞直到收到有效的 Boss 房狀態。

        有效條件：
          - in_boss_room == True
          - boss.hp > 0（Boss 已被 BossTracker 偵測到）
          - player.hp > 0（玩家還活著，避免一進去就已經死了）

        【為什麼需要這個等待？】
        TCP 連線建立後，Mod 會立刻推送一次 connected 事件，
        但玩家可能還沒進 Boss 房。如果不等待就直接開始，
        reset() 拿到的初始觀測可能是大廳的狀態，
        第一局的訓練資料會完全錯誤。
        """
        # 【兩階段等待】
        # 第一階段：等到 Boss 房狀態出現且 Boss HP > 0
        # 第二階段：再等 BOSS_STABLE_SECS 秒確認 HP 沒有異常跳動
        # 這樣可以跳過入場動畫期間短暫出現的臨時 HealthManager

        deadline = time.time() + timeout

        # 第一階段：等到基本條件成立
        while time.time() < deadline:
            state = self._client.get_latest_state()
            if (
                state
                and state.get("in_boss_room")
                and state["boss"]["hp"] > 0
                and state["player"]["hp"] > 0
            ):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"等待 Boss 房逾時（{timeout}s），"
                "請確認遊戲已進入 GG_Hornet_1 且 Mod 已載入。"
            )

        # 第二階段：等待 Boss HP 穩定（連續 BOSS_STABLE_SECS 秒沒有變化）
        # 【為什麼要等穩定？】
        # 某些 Boss 入場時有臨時物件的 HP 會在幾百毫秒內歸零消失。
        # 等到 HP 不再跳動，才是真正開戰的起點。
        # 【關鍵：穩定後必須確認 HP 仍 > 0】
        # 如果穩定在 0（臨時物件消失後沒有真正的 Boss），
        # 要回到第一階段重新等待，不能拿 HP=0 的狀態開始。
        while time.time() < deadline:
            stable_hp    = state["boss"]["hp"]
            stable_since = time.time()

            # 等這個 HP 值穩定 BOSS_STABLE_SECS 秒
            while time.time() < deadline:
                time.sleep(0.1)
                state   = self._client.get_latest_state()
                curr_hp = state["boss"]["hp"] if state else 0

                if curr_hp != stable_hp:
                    # HP 還在變動，重置計時
                    stable_hp    = curr_hp
                    stable_since = time.time()
                elif time.time() - stable_since >= BOSS_STABLE_SECS:
                    break  # 穩定了

            # 穩定後確認 HP 仍 > 0，否則重新等待第一階段
            if stable_hp > 0:
                break

            # 穩定在 0：等下一個有效 Boss 出現
            print("[HKEnv] Boss HP 穩定在 0，等待真正的 Boss 生成...")
            while time.time() < deadline:
                time.sleep(0.1)
                state = self._client.get_latest_state()
                if state and state.get("in_boss_room") and state["boss"]["hp"] > 0:
                    break
        else:
            raise TimeoutError("等待 Boss HP 穩定逾時。")

        print(
            f"[HKEnv] Boss 就緒：{state['boss']['name']}  "
            f"HP={state['boss']['hp']}/{state['boss']['max_hp']}"
        )
        return state
