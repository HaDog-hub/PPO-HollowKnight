# HKAIBridge Protocol — Phase 1

## 傳輸層
- TCP, port 11000, 僅限本機 (127.0.0.1)
- 編碼：UTF-8
- 訊息格式：Newline-Delimited JSON（每條訊息以 `\n` 結尾）

---

## Mod → Python（狀態推送）

### 觸發時機
- 小騎士 HP 變化（受傷 / 回血）
- Boss HP 變化（被打）
- Python 發送 `get_state` 指令時（type 為 `"response"`）

### 格式

```json
{
  "type": "event",
  "timestamp": 1741910400000,
  "scene": "GG_Hornet_1",
  "player": {
    "hp": 4,
    "max_hp": 5,
    "hp_pct": 0.8000
  },
  "boss": {
    "name": "Hornet Boss 1",
    "hp": 525,
    "max_hp": 700,
    "hp_pct": 0.7500
  }
}
```

### 欄位說明

| 欄位 | 類型 | 說明 |
|------|------|------|
| `type` | string | `"event"`（主動推送）或 `"response"`（回應查詢） |
| `timestamp` | int | Unix 毫秒時間戳 |
| `scene` | string | 當前場景名稱（尋神者模式以 `GG_` 開頭） |
| `player.hp` | int | 玩家當前血量（mask 數） |
| `player.max_hp` | int | 玩家最大血量 |
| `player.hp_pct` | float | 血量百分比 0.0~1.0，供泛化使用 |
| `boss.name` | string | Boss GameObject 名稱，`"none"` 表示無 Boss |
| `boss.hp` | int | Boss 當前 HP |
| `boss.max_hp` | int | Boss 最大 HP（首次偵測時快取） |
| `boss.hp_pct` | float | Boss 血量百分比 0.0~1.0 |

---

## Python → Mod（Phase 1 查詢指令）

```json
{"cmd": "get_state"}
```

Mod 收到後立即回傳一條 `type: "response"` 的狀態快照。

> **RL 訓練注意**：每個 step 請用 `client.get_latest_state()` 取得快取狀態。
> `request_state()` 會主動發送查詢指令，適合手動測試（如 `test_bridge.py`），訓練時不需要。

---

## Phase 2 預留指令（尚未實作）

```json
{"cmd": "reset"}
{"cmd": "goto_scene", "scene": "GG_Hornet_1"}
{"cmd": "set_player_hp", "value": 9}
```
