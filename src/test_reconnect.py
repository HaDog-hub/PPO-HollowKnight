"""
test_reconnect.py — 不需要開遊戲，驗證 HKBridgeClient 斷線重連邏輯

測試流程：
  1. 啟動一個假的 TCP Server
  2. Client 連線
  3. Server 主動關閉連線（模擬 Mod 崩潰 / TCP 斷線）
  4. 驗證 _running 變成 False、_latest_state 被清空
  5. 重啟 Server，呼叫 connect() 驗證成功重連

執行：
    python src/test_reconnect.py
"""

import json
import socket
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hk_client import HKBridgeClient

PORT = 11099  # 用不同的 port 避免和真實 Mod 衝突

FAKE_STATE = json.dumps({
    "type": "state",
    "scene": "GG_Hornet_1",
    "in_boss_room": True,
    "player": {"hp": 5, "max_hp": 5, "hp_pct": 1.0},
    "boss": {"name": "Hornet", "hp": 900, "max_hp": 900, "hp_pct": 1.0},
}) + "\n"


def run_server_once(stop_after_accept: threading.Event, accepted: threading.Event):
    """啟動 server，接受一個連線，發幾筆資料，然後關閉（模擬斷線）"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    srv.settimeout(5.0)
    accepted.set()  # 通知 server 已就緒
    try:
        conn, _ = srv.accept()
        # 送幾筆假資料
        for _ in range(3):
            conn.sendall(FAKE_STATE.encode())
            time.sleep(0.05)
        # 模擬斷線：server 主動關閉
        conn.close()
        print("[FakeServer] 主動關閉連線（模擬斷線）")
    except socket.timeout:
        print("[FakeServer] accept 超時")
    finally:
        srv.close()
        stop_after_accept.set()


def run_server_second(ready: threading.Event, got_client: threading.Event):
    """第二個 server，等待 client 重連後送一筆資料確認重連成功"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    srv.settimeout(5.0)
    ready.set()
    try:
        conn, _ = srv.accept()
        print("[FakeServer2] 重連成功！收到 client 連線")
        conn.sendall(FAKE_STATE.encode())
        time.sleep(0.2)
        conn.close()
        got_client.set()
    except socket.timeout:
        print("[FakeServer2] 重連 accept 超時（client 沒有重連）")
    finally:
        srv.close()


def main():
    passed = 0
    failed = 0

    # ── Test 1：斷線後 _running 應該變 False ──────────────────────────────────
    print("\n=== Test 1：斷線後 _running 應變 False ===")
    srv_done   = threading.Event()
    srv_ready  = threading.Event()
    t = threading.Thread(target=run_server_once, args=(srv_done, srv_ready), daemon=True)
    t.start()
    srv_ready.wait(timeout=3)

    client = HKBridgeClient(port=PORT)
    client.connect()
    print(f"  連線後 _running = {client._running}")

    srv_done.wait(timeout=3)   # 等 server 關閉
    time.sleep(0.3)            # 讓 recv_loop 有時間偵測到斷線

    if not client._running:
        print("  [PASS] _running == False ✓")
        passed += 1
    else:
        print("  [FAIL] _running 仍是 True，修法沒有生效")
        failed += 1

    # ── Test 2：斷線後 _latest_state 應該被清空 ───────────────────────────────
    print("\n=== Test 2：斷線後 _latest_state 應被清空 ===")
    if client.get_latest_state() is None:
        print("  [PASS] _latest_state == None ✓")
        passed += 1
    else:
        print("  [FAIL] _latest_state 還有舊數據")
        failed += 1

    # ── Test 3：呼叫 connect() 應該能重連 ────────────────────────────────────
    print("\n=== Test 3：呼叫 connect() 應能成功重連 ===")
    srv2_ready    = threading.Event()
    srv2_got      = threading.Event()
    t2 = threading.Thread(target=run_server_second, args=(srv2_ready, srv2_got), daemon=True)
    t2.start()
    srv2_ready.wait(timeout=3)

    try:
        client.connect()
        print(f"  重連後 _running = {client._running}")
    except Exception as e:
        print(f"  [FAIL] connect() 拋出例外：{e}")
        failed += 1
        client = None

    if client is not None:
        srv2_got.wait(timeout=3)
        time.sleep(0.3)
        if client._running:
            print("  [PASS] _running == True（重連成功）✓")
            passed += 1
        else:
            print("  [FAIL] 重連後 _running 仍是 False")
            failed += 1

        # ── Test 4：重連後能收到新資料 ────────────────────────────────────────
        print("\n=== Test 4：重連後 _latest_state 應有新數據 ===")
        time.sleep(0.3)
        if client.get_latest_state() is not None:
            print("  [PASS] 重連後收到新 state ✓")
            passed += 1
        else:
            print("  [FAIL] 重連後 _latest_state 仍是 None")
            failed += 1

        client.disconnect()

    # ── 結果 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"結果：{passed} passed，{failed} failed")
    if failed == 0:
        print("所有測試通過 — 重連邏輯正常")
    else:
        print("有測試失敗，請檢查修改")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
