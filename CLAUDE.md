# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based AI project targeting Hollow Knight. The environment is set up for computer vision and reinforcement learning using PyTorch.
**Core principle**: AI learns purely from screen pixels — no reading game memory. Only reward signals and visual observations.

## Environment

Python 3.10 virtual environment at `venv/`. Activate before running anything:

```bash
source venv/Scripts/activate   # Windows (bash/Git Bash)
# or
venv\Scripts\activate.bat      # Windows CMD
```

## Key Dependencies (Already Installed)

- **torch** 2.10.0 + **torchvision** 0.25.0 — neural networks and CV
- **Pillow** 12.1.1 — image capture/processing
- **numpy** 2.2.6 — numerical ops
- **stable-baselines3** — PPO training
- **cutie** — video object segmentation (installed at `Cutie/`, `pip install -e .`)
- **mss** — fast screen capture
- **pynput** — keyboard input simulation

## Development Commands

```bash
python src/your_script.py       # run any script
pip install <package>           # install into venv
pip freeze > requirements.txt   # freeze dependencies
```

## Architecture

```
Hollow Knight + HKAIBridge Mod (C#)
        ↕ TCP JSON (port 11000, newline-delimited)
  src/hk_client.py        — 背景執行緒接收 Mod 推送的 JSON 狀態
  src/screen_capture.py   — 截圖，回傳 (gray: 144×256×1, color: 144×256×3)
  src/cutie_extractor.py  — Cutie 物件追蹤，輸出 (144×256×2) 二值遮罩
  src/hk_action.py        — 鍵盤執行器（pynput），15 個動作，含失焦防呆
  src/hk_env.py           — Gymnasium Env，obs/action/reward，Boss 房等待
  src/train.py            — PPO 訓練入口（自動 fresh / resume）
  src/callbacks.py        — 每局統計 + TensorBoard 自訂指標
  src/check_env.py        — 不開遊戲驗證 HKEnv（含 reward 邏輯測試）
  src/test_bridge.py      — 驗證 HKAIBridge Mod TCP 連線
```

## Per-Boss 訓練目錄

```
checkpoints/{boss_name}/   # 模型 checkpoint
logs/{boss_name}/          # TensorBoard log
labels/{boss_name}/        # Cutie 標注資料（imgs/ + masks/）
```

切換 Boss：`python src/train.py hornet_2`

## HKAIBridge Mod

- TCP Server（port 11000），newline-delimited JSON
- `BossTracker`：偵測 Godhome Boss HealthManager，記錄 `LastBossScene`
- `PlayerTracker`：偵測玩家 HP 變化
- **自動重置**：第一局手動進入 Boss 房後，Mod 記錄場景，死亡或勝利後自動重播

## Hollow Knight 版本

**必須使用 1.5.78.11833**，新版本不支援 Mod API。

## Cutie 標注

- 物件 1 = 騎士，物件 2 = Hornet（或對應 Boss）
- 使用 `Cutie/interactive_demo.py` 標注新 Boss 的遮罩
- `CutieExtractor` 在 `HKEnv.__init__()` 初始化，每局 `reset()` 清除動態記憶
