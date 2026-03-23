"""
test_bridge.py — 驗證 HKAIBridge Mod 連線是否正常

使用方法：
    1. 先啟動 Hollow Knight（確認 Mod 已載入）
    2. 進入尋神者模式的任意 Boss 房
    3. 執行此腳本：python src/test_bridge.py
"""

import time
from hk_client import HKBridgeClient


def on_state_change(state: dict) -> None:
    p = state["player"]
    b = state["boss"]
    print(
        f"[{state['type']:8s}] "
        f"scene={state['scene']:25s} | "
        f"player={p['hp']}/{p['max_hp']} ({p['hp_pct']*100:.1f}%) | "
        f"boss={b['name']} {b['hp']}/{b['max_hp']} ({b['hp_pct']*100:.1f}%)"
    )


def main() -> None:
    client = HKBridgeClient()

    print("Connecting to HKAIBridge Mod...")
    try:
        client.connect()
    except ConnectionRefusedError:
        print("ERROR: 無法連線。請確認遊戲已啟動且 Mod 已載入。")
        return

    # 事件驅動：HP 變化時自動印出
    client.on_state_change(on_state_change)

    # 主動查詢一次當前狀態
    time.sleep(0.2)
    print("\n=== 主動查詢當前狀態 ===")
    state = client.request_state()
    if state:
        on_state_change(state)
    else:
        print("（尚未收到任何狀態，請先進入 Boss 房）")

    print("\n=== 等待 HP 變化事件（Ctrl+C 結束）===")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
