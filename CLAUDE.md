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
# Run any script
python src/your_script.py

# Install new packages into the venv
pip install <package>

# Freeze current dependencies
pip freeze > requirements.txt
```

## Architecture Notes

### Pipeline

```
Hollow Knight + HKAIBridge Mod (C#)
        ↕ TCP JSON (port 11000, newline-delimited)
  src/hk_client.py        — 接收遊戲狀態（HP、場景、in_boss_room）
        ↓
  src/screen_capture.py   — 截圖，回傳灰階 (144,256,1) + 彩色 (144,256,3)
  src/cutie_extractor.py  — Cutie 物件追蹤，回傳遮罩 (144,256,2)
        ↓
  src/hk_env.py           — Gymnasium 環境，obs/action/reward，Boss 房等待
        ↓
  src/train.py            — SB3 PPO 訓練主腳本（MultiInputPolicy）
        ↓
  src/hk_action.py        — 模擬鍵盤輸入（pynput），15 種離散動作，每步 100ms
```

### Source Files

| 檔案 | 用途 |
|------|------|
| `src/hk_client.py` | TCP client，背景執行緒接收 Mod 推送的 JSON 狀態 |
| `src/hk_action.py` | 動作定義表 + 鍵盤執行器（pynput），15 個動作 |
| `src/screen_capture.py` | mss 截圖，回傳 `(gray, color)`：灰階 (144,256,1) 和彩色 (144,256,3) RGB |
| `src/cutie_extractor.py` | Cutie 物件追蹤封裝，輸出 (144,256,2) 二值遮罩（騎士+Boss） |
| `src/hk_env.py` | Gymnasium `Env`，obs/action/reward 定義，Boss 房等待 |
| `src/train.py` | PPO 訓練入口，per-boss checkpoint + TensorBoard |
| `src/resume_train.py` | 從 checkpoint 繼續訓練，per-boss 目錄 |
| `src/callbacks.py` | SB3 自訂 callback，每局統計 + TensorBoard 自訂指標 |
| `src/check_env.py` | 不開遊戲驗證 HKEnv 符合 Gym 規範（含 reward 邏輯測試） |
| `src/test_bridge.py` | 驗證 HKAIBridge Mod TCP 連線 |

### Observation Space

```python
Dict({
    "screen": Box(144, 256, 10, uint8),
    # channel 0-7：最近 8 幀灰階（frame stacking）
    # channel 8：  騎士分割遮罩（0 或 255）
    # channel 9：  Hornet 分割遮罩（0 或 255）

    "stats": Box(2,, float32),
    # [player_hp_pct, boss_hp_pct]
})
```

### Action Space

15 個離散動作（`src/hk_action.py`）：
- 基本移動：左、右、跳、二段跳
- 攻擊：普攻、上劈、下劈（下劈僅在空中有效）
- 法術：靈魂爆發等
- IDLE：不做任何輸入

### Cutie 物件追蹤

- 標注資料：`labels/hornet/imgs/`（截圖）、`labels/hornet/masks/`（indexed 遮罩）
- 物件 1 = 騎士，物件 2 = Hornet
- `CutieExtractor` 在 `HKEnv.__init__()` 初始化，每局呼叫 `reset()`
- 使用 Cutie GUI（`Cutie/interactive_demo.py`）標注新 Boss 的遮罩

### Per-Boss 訓練目錄

```
checkpoints/{boss_name}/   # 模型 checkpoint
logs/{boss_name}/          # TensorBoard log
labels/{boss_name}/        # Cutie 標注資料
```

切換 Boss：
```bash
python src/train.py hornet_1     # 預設
python src/train.py hornet_2
python src/resume_train.py hornet_1
```

### HKAIBridge Mod 架構

- TCP Server（port 11000），newline-delimited JSON
- `BossTracker`：偵測 Godhome Boss HealthManager，記錄 `LastBossScene`
- `PlayerTracker`：偵測玩家 HP 變化
- **自動重置**：第一局手動進入 Boss 房後，Mod 記錄場景參數，死亡或勝利後自動重播

### Hollow Knight 版本

**必須使用 1.5.78.11833**，新版本不支援 Mod API。

### Known TODOs

- Step 3：修改 `hk_env.py`（FRAME_STACK 4→8，加入 Cutie，更新 obs space）
- Step 4：修改 `train.py`（從頭訓練設定）
- 訓練完成後評估效果，考慮擴展到其他 Boss
