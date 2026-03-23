"""
hk_client.py — HKAIBridge TCP Client

負責與 Mod 通訊，提供兩種使用方式：

1. 被動監聽（事件驅動）：
       client.on_state_change(callback)

2. 主動查詢（RL 每步呼叫）：
       state = client.get_latest_state()

訊息格式：newline-delimited JSON，每條以 '\\n' 結尾
"""

import json
import socket
import threading
from typing import Callable, Optional


class HKBridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11000):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._latest_state: Optional[dict] = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[dict], None]] = []
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    # ── 連線管理 ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """連接到 Mod 的 TCP Server，啟動背景接收執行緒"""
        if self._running and self._sock is not None:
            return  # 已連線，不重複建立
        # 清掉舊的 socket（斷線後重連時需要）
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._host, self._port))
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="HKBridge-Recv"
        )
        self._recv_thread.start()
        print(f"[HKBridgeClient] Connected to Mod at {self._host}:{self._port}")

    def disconnect(self) -> None:
        self._running = False
        if self._sock:
            self._sock.close()

    # ── 接收迴圈（背景執行緒）────────────────────────────────────────────────

    def _recv_loop(self) -> None:
        """持續接收 Mod 推送的 JSON 事件，更新 latest_state 並呼叫 callbacks"""
        buf = ""
        try:
            while self._running:
                try:
                    chunk = self._sock.recv(4096).decode("utf-8")  # type: ignore[union-attr]
                    if not chunk:
                        break
                    buf += chunk

                    # newline-delimited JSON：每行一條訊息
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            state = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        with self._lock:
                            self._latest_state = state

                        for cb in self._callbacks:
                            try:
                                cb(state)
                            except Exception as e:
                                print(f"[HKBridgeClient] Callback error: {e}")

                except Exception as e:
                    if self._running:
                        print(f"[HKBridgeClient] Recv error: {e}")
                    break
        finally:
            # 確保 disconnect 後 _running=False，讓 connect() 能重新建立連線
            self._running = False
            with self._lock:
                self._latest_state = None  # 清掉舊 cache，避免重連後用到過期數據
            print("[HKBridgeClient] Disconnected from Mod")

    # ── 公開 API ──────────────────────────────────────────────────────────────

    def on_state_change(self, callback: Callable[[dict], None]) -> None:
        """註冊 HP 變化回呼（事件驅動模式）"""
        self._callbacks.append(callback)

    def get_latest_state(self) -> Optional[dict]:
        """取得最新快取狀態（RL step 模式，直接讀取，不需等待事件）"""
        with self._lock:
            return self._latest_state

    def request_state(self) -> Optional[dict]:
        """主動向 Mod 查詢當前狀態，等待回應後回傳（同步）"""
        if not self._sock:
            return None
        try:
            self._sock.sendall(b'{"cmd":"get_state"}\n')
        except Exception as e:
            print(f"[HKBridgeClient] Send error: {e}")
        return self.get_latest_state()

