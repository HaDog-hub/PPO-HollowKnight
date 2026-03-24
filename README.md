# Hollow Knight AI

用強化學習（PPO）訓練 AI 從純視覺（螢幕截圖）學習打贏 Hollow Knight 的 Boss。
不讀遊戲記憶體，只靠畫面和 reward 訊號學習。

**目前目標：** 尋神者模式 — Hornet Protector（`GG_Hornet_1`）

---

## 架構

```
Hollow Knight 遊戲
        ↕  TCP port 11000（Newline-Delimited JSON）
  mod/HKAIBridge              ← C# Mod，推送遊戲狀態 + 自動重置
        ↕
  src/hk_client.py            ← Python TCP Client
  src/screen_capture.py       ← 截圖（灰階 + 彩色）
  src/cutie_extractor.py      ← Cutie 物件追蹤，輸出騎士/Boss 遮罩
  src/hk_action.py            ← 鍵盤輸入模擬，15 個離散動作
        ↕
  src/hk_env.py               ← Gymnasium 環境封裝
        ↕
  src/train.py                ← PPO 訓練主腳本（SB3）
```

---

## 環境需求

- Python 3.10
- Hollow Knight 版本 **1.5.78.11833**（新版不支援 Mod API）
- GPU 建議（RTX 5060 Ti 或同等）
- [Hollow Knight Modding API](https://github.com/hk-modding/api)

---

## 安裝

```bash
# 啟動虛擬環境
source venv/Scripts/activate      # Windows Git Bash
venv\Scripts\activate.bat         # Windows CMD

# 安裝依賴
pip install -r requirements.txt

# 安裝 Cutie（物件追蹤）
cd Cutie && pip install -e . && python cutie/utils/download_models.py && cd ..
```

### Mod 安裝（HKAIBridge）

1. 確認已安裝 Hollow Knight Modding API
2. 將 `mod/HKAIBridge/bin/Debug/net472/HKAIBridge.dll` 複製到遊戲的 `Mods/` 資料夾
3. 啟動遊戲，Mod 自動在 `127.0.0.1:11000` 開啟 TCP Server
4. 第一局需手動走進 Boss 房，之後 Mod 自動重置場景

---

## 訓練

```bash
python src/train.py              # 自動從頭或接續最新 checkpoint
python src/train.py --fresh      # 強制從頭
python src/train.py hornet_2     # 指定 Boss

tensorboard --logdir logs/hornet_1/
```

---

## Observation Space

```
screen : (144, 256, 10)  uint8
  channel 0-7 : 最近 8 幀灰階畫面（frame stacking）
  channel 8   : 騎士分割遮罩（Cutie 輸出）
  channel 9   : Hornet 分割遮罩（Cutie 輸出）

stats  : (2,)  float32   [player_hp_pct, boss_hp_pct]
```

---

## Reward 設計

每步 dense reward + Boss HP 里程碑 + 終局獎懲。
詳細數值見 `PROGRESS.md` 或 `src/hk_env.py`。

---

## 訓練進度

見 `PROGRESS.md`。
