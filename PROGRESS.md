# Hollow Knight AI 專案進度記錄

## 專案目標
用強化學習（PPO）訓練 AI 從純視覺（螢幕截圖）學習打敗 Hollow Knight 的 Boss。
不讀取遊戲記憶體，只靠畫面和簡單的 reward 訊號。

最終目標：打敗多數 Boss，具泛化能力
中期目標：打敗 GG_Hornet_1（尋神者模式）

---

## 訓練歷史

| 步數 | 事件 |
|------|------|
| 0 | 初期建置完成（Mod、HKEnv、PPO、自動重置） |
| ~700k | boss_dmg_pct 第一次平台期（約 30~40%） |
| ~961k | ent_coef 0.01→0.03，短暫下降後回升 |
| ~1.2M | 緩步上升，確認 entropy 調整有效 |
| ~1.46M | 再次平台期，boss_dmg_pct 穩定在 50~60% |
| ~1.46M | 細化 reward 里程碑 + ent_coef 降回 0.02 |
| 當前 | Cutie 整合完成（Step 1~4），準備從頭重新訓練 |

---

## 目前超參數

```python
LEARNING_RATE = 1e-4
ENT_COEF      = 0.02
N_STEPS       = 512
BATCH_SIZE    = 64
N_EPOCHS      = 5
FRAME_STACK   = 4   # 即將改為 8
```

### Reward 結構（最新版）
```python
REWARD_BOSS_DMG   = +5.0
REWARD_PLAYER_DMG = -3.0
REWARD_TIME_STEP  = -0.01
REWARD_SURVIVAL   = +0.005
REWARD_WIN        = +20.0
REWARD_LOSE       = -20.0

REWARD_MILESTONES = {
    0.75: +3.0,
    0.65: +3.0,
    0.55: +4.0,
    0.45: +5.0,
    0.35: +6.0,
    0.25: +8.0,
    0.15: +10.0,
    0.05: +12.0,
}
```

---

## 架構改動進度（目前進行中）

引入 Cutie 物件追蹤，將騎士和 Hornet 的分割遮罩加進 obs space。
此改動需要重新訓練（obs 結構改變，舊 checkpoint 不相容）。

| Step | 內容 | 狀態 |
|------|------|------|
| 1 | `screen_capture.py` 同時回傳灰階 + 彩色 | 完成 |
| 2 | `cutie_extractor.py` Cutie 封裝類別 | 完成 |
| 3 | `hk_env.py` FRAME_STACK 4→8，加入遮罩 obs | 完成 |
| 4 | `train.py` 從頭訓練設定 | 完成 |

### obs space 變更（Step 3 後）
```
screen: (144, 256, 4)  →  (144, 256, 10)
         4幀灰階            8幀灰階 + 騎士遮罩 + Hornet遮罩
stats:  (2,) 不變
```

---

## 關鍵技術決策記錄

- **Cutie 向量 vs 遮罩**：選擇遮罩 channel 而非原始 2048 維向量。
  理由：2048 維對 PPO 的 MLP 太大難以學習；遮罩是空間資訊，CNN 天生擅長處理。

- **FRAME_STACK 4→8**：RTX 5060 Ti 有足夠算力，且都要重新訓練了，直接用最大收益設定。

- **不整個遷移到 OC-STORM**：OC-STORM 開源了，但直接用等於用別人的東西，鑑別度低。
  現在的做法是引用 Cutie 工具、自己整合進 PPO 架構，保留專案獨立性。

---

## 標注資料
- 路徑：`labels/hornet/imgs/`（7張截圖）、`labels/hornet/masks/`（7張遮罩）
- 物件 1 = 騎士，物件 2 = Hornet
- 使用 Cutie GUI（`interactive_demo.py`）標注，workspace/masks/ 為 indexed 格式

---

## 硬體環境
- GPU：RTX 5060 Ti
- HK 版本：1.5.78.11833
- Python 3.10，venv 在 `venv/`
- Cutie 安裝在 `Cutie/`，OC-STORM 在 `OC-STORM/`
