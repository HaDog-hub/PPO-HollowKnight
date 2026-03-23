# Hollow Knight AI

用強化學習（PPO）訓練 AI 從純視覺（螢幕截圖）學習打贏 Hollow Knight 的 Boss。
不讀遊戲記憶體，只靠畫面和 reward 訊號學習。

**目前目標：** 尋神者模式（Hall of Gods）— Hornet Protector（`GG_Hornet_1`）

---

## 架構總覽

```
Hollow Knight 遊戲
        ↕  TCP port 11000（Newline-Delimited JSON）
  mod/HKAIBridge              ← C# Mod，推送遊戲狀態 + 自動重置
        ↕
  src/hk_client.py            ← Python TCP Client，接收狀態
  src/screen_capture.py       ← 截圖（灰階 + 彩色）
  src/cutie_extractor.py      ← Cutie 物件追蹤，輸出遮罩
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
- GPU（RTX 5060 Ti 或同等）
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
cd Cutie
pip install -e .
python cutie/utils/download_models.py
cd ..
```

---

## Mod 安裝（HKAIBridge）

1. 確認已安裝 Hollow Knight Modding API
2. 將 `mod/HKAIBridge/bin/Debug/net472/HKAIBridge.dll` 複製到遊戲的 `Mods/` 資料夾
3. 啟動遊戲，Mod 會自動在 `127.0.0.1:11000` 開啟 TCP Server
4. 第一局需手動走進 Boss 房，之後 Mod 自動重置場景

---

## 專案結構

```
HollowKnight_AI/
├── mod/                        C# Mod 原始碼（HKAIBridge）
├── src/
│   ├── hk_client.py            TCP Client
│   ├── hk_action.py            鍵盤輸入（15 個動作）
│   ├── screen_capture.py       截圖（回傳灰階 + 彩色）
│   ├── cutie_extractor.py      Cutie 物件追蹤封裝
│   ├── hk_env.py               Gymnasium 環境
│   ├── callbacks.py            訓練監控 Callback
│   ├── train.py                訓練主腳本
│   ├── resume_train.py         從 checkpoint 繼續訓練
│   ├── check_env.py            環境驗證（不需開遊戲）
│   └── test_bridge.py          Mod 連線測試
├── labels/
│   └── hornet/
│       ├── imgs/               Cutie 標注截圖（7 張）
│       └── masks/              Cutie indexed 遮罩
├── Cutie/                      Cutie 原始碼（pip install -e .）
├── OC-STORM/                   OC-STORM 參考實作（參考用）
├── checkpoints/                模型存檔（自動產生）
├── logs/                       TensorBoard 日誌（自動產生）
├── PROGRESS.md                 詳細訓練進度記錄
└── README.md
```

---

## Observation Space

```python
Dict({
    "screen": Box(144, 256, 10, uint8),
    # channel 0-7：最近 8 幀灰階畫面（frame stacking）
    # channel 8：  騎士分割遮罩（Cutie 輸出，0 或 255）
    # channel 9：  Hornet 分割遮罩（Cutie 輸出，0 或 255）

    "stats": Box(2,, float32),
    # [player_hp_pct, boss_hp_pct]
})
```

---

## Action Space

15 個離散動作（`src/hk_action.py`）：
移動、跳躍、二段跳、普攻、上劈、下劈、衝刺、法術、IDLE 等組合。

---

## Reward 結構

```python
# 每步 dense reward
boss_dmg   × +5.0    # 打到 Boss
player_dmg × -3.0    # 被打
time_step    -0.01   # 時間懲罰
hp_pct     × +0.005  # 存活獎勵

# 里程碑（每局每個門檻只觸發一次）
0.75 HP → +3.0
0.65 HP → +3.0
0.55 HP → +4.0
0.45 HP → +5.0
0.35 HP → +6.0
0.25 HP → +8.0
0.15 HP → +10.0
0.05 HP → +12.0

# 終局
WIN  → +20.0
LOSE → -20.0
```

---

## 訓練流程

```bash
# 第一次訓練（從頭開始）
python src/train.py

# 從 checkpoint 繼續
python src/resume_train.py

# 監控訓練
tensorboard --logdir logs/hornet_1/
```

---

## 訓練進度

詳見 `PROGRESS.md`。
