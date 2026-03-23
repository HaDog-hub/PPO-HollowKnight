"""
callbacks.py — 自訂 SB3 訓練 Callback

提供 EpisodeLogCallback：
  - 每局結束後在終端機印出可讀的摘要
  - 同步把自訂指標送進 TensorBoard

【什麼是 SB3 Callback？】
SB3 在訓練過程中會在固定時機呼叫 Callback 的方法，讓你「插入」
自己的邏輯，而不需要修改訓練主迴圈。主要時機有：

  _on_step()       : 每一步之後（最常用）
  _on_rollout_end(): 每次收集完 n_steps 步之後（更新模型前）
  _on_training_end(): 整個訓練結束後

這支腳本使用 _on_step() 追蹤每局的統計，
在 dones=True（局結束）時印出摘要並送出 TensorBoard 資料。
"""

import os
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """
    每局結束後印出摘要 + 送出 TensorBoard 指標。

    追蹤的指標：
      episode/reward          每局總 reward
      episode/steps           每局存活步數（越長越好，代表沒有馬上死）
      episode/boss_dmg_pct    對 Boss 造成的總傷害百分比（0.0~1.0）
      episode/win_rate        最近 N 局的勝率（滾動平均）
    """

    def __init__(self, window: int = 10, verbose: int = 0, episode_count_path: str = None):
        """
        Args:
            window: 計算滾動平均用的局數窗口（預設最近 10 局）
            episode_count_path: 累計局數的持久化檔案路徑，None 表示不持久化
        """
        super().__init__(verbose)

        self._window = window
        self._episode_count_path = episode_count_path

        # 當前局的累計值
        self._ep_reward: float = 0.0
        self._ep_steps:  int   = 0
        self._ep_start_boss_hp: float = 1.0   # 這局開始時 Boss 的 hp_pct

        # 統計歷史（用 deque 自動維持固定長度）
        # 【為什麼用 deque 而不是 list？】
        # deque(maxlen=N) 在達到上限時會自動丟掉最舊的資料，
        # 不需要手動 .pop(0)，也比 list 在頭部刪除更有效率。
        self._history: deque = deque(maxlen=window)

        # 從檔案讀取累計局數，沒有檔案就從 0 開始
        self._total_episodes: int = self._load_episode_count()
        self._best_reward: float = float("-inf")

    # ── 持久化局數 ────────────────────────────────────────────────────────────

    def _load_episode_count(self) -> int:
        if self._episode_count_path and os.path.exists(self._episode_count_path):
            with open(self._episode_count_path, "r") as f:
                return int(f.read().strip())
        return 0

    def _save_episode_count(self) -> None:
        if self._episode_count_path:
            with open(self._episode_count_path, "w") as f:
                f.write(str(self._total_episodes))

    # ── SB3 介面 ─────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        """
        每步都被呼叫。

        【self.locals 是什麼？】
        SB3 在呼叫 _on_step 前會把這一步的資料放進 self.locals：
          rewards : 這步的 reward（list，因為可能有多個平行環境）
          dones   : 這步是否結束（list）
          infos   : 我們在 hk_env.step() 回傳的 info dict（list）

        我們只有一個環境，所以取 index [0]。
        """
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        info   = self.locals["infos"][0]

        # 累計這局的數值
        self._ep_reward += reward
        self._ep_steps  += 1

        # 用來算 boss 傷害：記下這局開始時的 boss hp
        if self._ep_steps == 1:
            self._ep_start_boss_hp = info.get("boss_hp_pct", 1.0)

        # 局結束時處理
        if done:
            self._on_episode_end(info)

        return True  # 回傳 False 可以提早停止訓練，這裡永遠回傳 True

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _on_episode_end(self, info: dict) -> None:
        """計算本局統計、印出摘要、送出 TensorBoard 資料"""

        self._total_episodes += 1
        self._save_episode_count()

        # 判斷結果
        # 【為什麼用 get() 而不是直接取 info["boss_hp"]？】
        # disconnect 情況下 info 只有 {"reason": "disconnect"}，
        # 直接取 key 會 KeyError，用 get() 搭配預設值更安全。
        boss_hp_pct_end = info.get("boss_hp_pct", self._ep_start_boss_hp)
        boss_dmg_pct    = self._ep_start_boss_hp - boss_hp_pct_end
        boss_dmg_pct    = max(0.0, min(1.0, boss_dmg_pct))  # 夾在 [0,1]

        is_win = info.get("boss_hp", 1) <= 0

        # 更新歷史
        self._history.append({
            "reward":   self._ep_reward,
            "won":      is_win,
            "boss_dmg": boss_dmg_pct,
            "steps":    self._ep_steps,
        })

        if self._ep_reward > self._best_reward:
            self._best_reward = self._ep_reward

        # 滾動統計
        win_rate   = sum(h["won"]      for h in self._history) / len(self._history)
        avg_dmg    = sum(h["boss_dmg"] for h in self._history) / len(self._history)
        avg_reward = sum(h["reward"]   for h in self._history) / len(self._history)

        # 印出本局摘要
        self._print_episode(is_win, boss_dmg_pct)

        # 每 window 局印一次滾動統計
        if self._total_episodes % self._window == 0:
            self._print_rolling_stats(win_rate, avg_dmg, avg_reward)

        # 送出 TensorBoard 自訂指標
        # 【self.logger.record() 是什麼？】
        # SB3 的 logger 會把這些值傳給 TensorBoard（以及終端機 verbose log）。
        # key 格式用 "分類/名稱"，TensorBoard 會自動分組顯示。
        # 【為什麼要檢查 self.model？】
        # logger 是透過 self.model 取得的，在真實訓練時一定存在。
        # 但在獨立測試（check_env / 單元測試）時 model 還沒被注入，
        # 加上 hasattr 保護讓這個 class 在兩種情境都能正常使用。
        if hasattr(self, "model") and self.model is not None:
            self.logger.record("episode/reward",       self._ep_reward)
            self.logger.record("episode/steps",        self._ep_steps)
            self.logger.record("episode/boss_dmg_pct", boss_dmg_pct)
            self.logger.record("episode/win_rate",     win_rate)
            self.logger.record("episode/is_win",       float(is_win))

        # 重置本局計數器
        self._ep_reward        = 0.0
        self._ep_steps         = 0
        self._ep_start_boss_hp = 1.0

    def _print_episode(self, is_win: bool, boss_dmg_pct: float) -> None:
        """印出單局摘要（含 Boss 傷害的 ASCII 進度條）"""

        # ASCII 進度條：10 格，每格代表 10% 傷害
        filled = int(boss_dmg_pct * 10)
        bar    = "#" * filled + "-" * (10 - filled)

        result_str = " WIN! " if is_win else " LOSE "

        print(
            f"[Ep {self._total_episodes:04d}]  "
            f"Steps: {self._ep_steps:4d}  |  "
            f"Reward: {self._ep_reward:+8.2f}  |  "
            f"Boss: [{bar}] {boss_dmg_pct*100:5.1f}%  |"
            f"{result_str}"
        )

    def _print_rolling_stats(
        self, win_rate: float, avg_dmg: float, avg_reward: float
    ) -> None:
        """印出最近 N 局的滾動統計"""
        sep = "-" * 70
        print(sep)
        print(
            f"  最近 {len(self._history)} 局  |  "
            f"勝率: {win_rate*100:.0f}%  |  "
            f"平均 Boss 傷害: {avg_dmg*100:.1f}%  |  "
            f"平均 Reward: {avg_reward:+.2f}  |  "
            f"最佳 Reward: {self._best_reward:+.2f}"
        )
        print(sep)
        print()
