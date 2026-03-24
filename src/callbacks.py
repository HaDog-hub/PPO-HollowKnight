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
      episode/steps           每局存活步數
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
        self._ep_start_boss_hp: float = 1.0

        # 統計歷史（deque 自動維持固定長度）
        self._history: deque = deque(maxlen=window)

        # 從檔案讀取累計局數
        self._total_episodes: int = self._load_episode_count()
        self._best_reward:    float = float("-inf")
        self._best_boss_dmg:  float = 0.0

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
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        info   = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_steps  += 1

        if self._ep_steps == 1:
            self._ep_start_boss_hp = info.get("boss_hp_pct", 1.0)

        if done:
            self._on_episode_end(info)

        return True

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _on_episode_end(self, info: dict) -> None:
        self._total_episodes += 1
        self._save_episode_count()

        boss_hp_pct_end = info.get("boss_hp_pct", self._ep_start_boss_hp)
        boss_dmg_pct    = max(0.0, min(1.0, self._ep_start_boss_hp - boss_hp_pct_end))
        player_hp_end   = info.get("player_hp", 0)
        boss_hp_end     = info.get("boss_hp", 1)

        # 判斷結果
        if boss_hp_end <= 0:
            result = "win"
        elif player_hp_end <= 0:
            result = "lose"
        else:
            result = "trunc"   # 卡關 / 離開 Boss 房 / 超時

        # 更新歷史與最佳紀錄
        self._history.append({
            "reward":   self._ep_reward,
            "won":      result == "win",
            "boss_dmg": boss_dmg_pct,
            "steps":    self._ep_steps,
        })

        if self._ep_reward > self._best_reward:
            self._best_reward = self._ep_reward
        if boss_dmg_pct > self._best_boss_dmg:
            self._best_boss_dmg = boss_dmg_pct

        # 滾動統計
        win_rate   = sum(h["won"]      for h in self._history) / len(self._history)
        avg_dmg    = sum(h["boss_dmg"] for h in self._history) / len(self._history)
        avg_reward = sum(h["reward"]   for h in self._history) / len(self._history)
        avg_steps  = sum(h["steps"]    for h in self._history) / len(self._history)

        self._print_episode(result, boss_dmg_pct, player_hp_end)

        if self._total_episodes % self._window == 0:
            self._print_rolling_stats(win_rate, avg_dmg, avg_reward, avg_steps)

        if hasattr(self, "model") and self.model is not None:
            self.logger.record("episode/reward",       self._ep_reward)
            self.logger.record("episode/steps",        self._ep_steps)
            self.logger.record("episode/boss_dmg_pct", boss_dmg_pct)
            self.logger.record("episode/win_rate",     win_rate)
            self.logger.record("episode/is_win",       float(result == "win"))

        self._ep_reward        = 0.0
        self._ep_steps         = 0
        self._ep_start_boss_hp = 1.0

    def _print_episode(self, result: str, boss_dmg_pct: float, player_hp: int) -> None:
        """印出單局摘要"""

        # Boss 傷害進度條（20 格，每格 5%）
        filled = int(boss_dmg_pct * 20)
        bar    = "█" * filled + "░" * (20 - filled)

        result_tag = {"win": "* WIN ", "lose": "  LOSE", "trunc": " TRUNC"}[result]

        timesteps = self.num_timesteps if hasattr(self, "num_timesteps") else 0

        print(
            f"[Ep {self._total_episodes:04d}]  "
            f"Steps {self._ep_steps:4d}  |  "
            f"Player HP {player_hp}  |  "
            f"Boss [{bar}] {boss_dmg_pct*100:5.1f}%  |  "
            f"Reward {self._ep_reward:+8.2f}  |  "
            f"{result_tag}  "
            f"(total {timesteps:,} steps)"
        )

    def _print_rolling_stats(
        self, win_rate: float, avg_dmg: float, avg_reward: float, avg_steps: float
    ) -> None:
        """印出最近 N 局的滾動統計"""
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  Last {len(self._history)} episodes  (total {self._total_episodes} eps)")
        print(f"  {'Win Rate':<16}: {win_rate*100:5.1f}%      {'Best Boss Dmg':<16}: {self._best_boss_dmg*100:5.1f}%")
        print(f"  {'Avg Boss Dmg':<16}: {avg_dmg*100:5.1f}%      {'Avg Steps':<16}: {avg_steps:6.0f}")
        print(f"  {'Avg Reward':<16}: {avg_reward:+8.2f}   {'Best Reward':<16}: {self._best_reward:+8.2f}")
        print(f"{sep}\n")
