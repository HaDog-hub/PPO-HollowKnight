"""
train.py — PPO 訓練主腳本

自動偵測是否有 checkpoint，有則繼續，無則從頭訓練。
加 --fresh 強制從頭。

使用方式：
    python src/train.py                    # 預設 hornet_1，自動從頭或繼續
    python src/train.py hornet_2           # 指定 Boss
    python src/train.py hornet_1 --fresh   # 強制從頭重新訓練

監控訓練：
    tensorboard --logdir logs/<BOSS_NAME>/
    瀏覽器開啟 http://localhost:6006
"""

import os
import sys
import glob
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from hk_env import HKEnv
from callbacks import EpisodeLogCallback


# ── Boss 設定 ─────────────────────────────────────────────────────────────────
BOSS_NAME   = next((a for a in sys.argv[1:] if not a.startswith("--")), "hornet_1")
FORCE_FRESH = "--fresh" in sys.argv

# ── 超參數 ───────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000  # 實際以 Ctrl+C 停止；改成 10_000 可快速驗證流程

N_STEPS       = 512
BATCH_SIZE    = 64
N_EPOCHS      = 5
LEARNING_RATE = 1e-4
ENT_COEF      = 0.02

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
LOG_DIR        = f"logs/{BOSS_NAME}/"
CHECKPOINT_DIR = f"checkpoints/{BOSS_NAME}/"
FINAL_MODEL    = f"checkpoints/{BOSS_NAME}/hk_ppo_final"

if FORCE_FRESH and os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    print(f"舊 log 已清除：{LOG_DIR}")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def find_latest_checkpoint() -> str | None:
    """找最新的 *steps.zip checkpoint，沒有則回傳 None"""
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "hk_ppo_*steps.zip"))
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("_")[-2]))
    return files[-1].replace(".zip", "")


# ── 建立環境 ──────────────────────────────────────────────────────────────────
env = HKEnv()

# ── 建立或載入模型 ────────────────────────────────────────────────────────────
checkpoint = None if FORCE_FRESH else find_latest_checkpoint()

if checkpoint:
    print(f"找到 checkpoint，繼續訓練：{checkpoint}")
    model = PPO.load(
        checkpoint,
        env=env,
        device="auto",
        tensorboard_log=LOG_DIR,
        # custom_objects 覆蓋 checkpoint 裡儲存的超參數，
        # 讓繼續訓練時的設定與這裡定義的一致
        custom_objects={
            "learning_rate": LEARNING_RATE,
            "n_epochs":      N_EPOCHS,
            "ent_coef":      ENT_COEF,
        },
    )
    reset_num_timesteps = False
else:
    reason = "--fresh 指定" if FORCE_FRESH else "沒有 checkpoint"
    print(f"{reason}，從頭開始訓練。")

    # 【為什麼用 MultiInputPolicy？】
    # observation_space 是 Dict（screen + stats），
    # MultiInputPolicy 會自動對 screen 用 CNN、對 stats 用 MLP，
    # 最後把兩個特徵向量接在一起送進 policy/value head。
    #
    # 【obs 結構（channel 0~9）】
    # channel 0~7：最近 8 幀灰階截圖（frame stacking，CNN 感知動態）
    # channel 8  ：騎士分割遮罩（Cutie 輸出）
    # channel 9  ：Hornet 分割遮罩（Cutie 輸出）
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )
    reset_num_timesteps = True

print("=" * 55)
print(f"  Hollow Knight AI — PPO 訓練（{BOSS_NAME}）")
print("=" * 55)
print(f"  模式       : {'從頭' if reset_num_timesteps else '繼續'}")
print(f"  已訓練步數 : {model.num_timesteps:,}")
print(f"  模型裝置   : {model.device}")
print(f"  Checkpoint : {CHECKPOINT_DIR}")
print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
print("=" * 55)
print(f"\n網路架構：\n{model.policy}\n")


# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=5_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="hk_ppo",
    verbose=0,
)

episode_log_cb = EpisodeLogCallback(
    window=10,
    episode_count_path=os.path.join(CHECKPOINT_DIR, "episode_count.txt"),
)


# ── 開始訓練 ──────────────────────────────────────────────────────────────────
print("開始訓練！腳本正在等待遊戲連線...")
print("請確認：")
print("  1. Hollow Knight 已開啟")
print("  2. HKAIBridge Mod 已載入")
print("  3. 已進入尋神者模式大廳\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([checkpoint_cb, episode_log_cb]),
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=f"PPO_{BOSS_NAME}",
    )
except KeyboardInterrupt:
    print("\n訓練被中斷（Ctrl+C），儲存當前模型...")

finally:
    model.save(FINAL_MODEL)
    env.close()
    print(f"模型已儲存至：{FINAL_MODEL}.zip")
    print("訓練結束。")
