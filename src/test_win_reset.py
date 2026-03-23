"""
test_win_reset.py — 手動測試勝利重置流程

用途：
    不啟動 AI，讓玩家自己打大黃蜂，
    驗證勝利後 Mod 是否正確重載 Boss 房。

使用方式：
    1. 開遊戲，載入 HKAIBridge Mod
    2. 手動走進 GG_Hornet_1
    3. 執行此腳本（另一個終端機）：
           python src/test_win_reset.py
    4. 自己打大黃蜂，觀察腳本輸出
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hk_client import HKBridgeClient

HOST = "127.0.0.1"
PORT = 11000

BOSS_STABLE_SECS  = 1.0
WAIT_TIMEOUT      = 120.0
MIN_BOSS_HP       = 1        # hp > 0 即視為有效


def wait_for_boss_room(client: HKBridgeClient) -> dict:
    """等到 Boss 房就緒（in_boss_room=True, boss.hp>0, player.hp>0）"""
    print("[test] 等待 Boss 房就緒...")
    deadline = time.time() + WAIT_TIMEOUT

    # 第一階段：等基本條件成立
    while time.time() < deadline:
        state = client.get_latest_state()
        if (
            state
            and state.get("in_boss_room")
            and state["boss"]["hp"] > 0
            and state["player"]["hp"] > 0
        ):
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("等待 Boss 房逾時")

    # 第二階段：等 Boss HP 穩定 1 秒
    while time.time() < deadline:
        stable_hp    = state["boss"]["hp"]
        stable_since = time.time()

        while time.time() < deadline:
            time.sleep(0.1)
            state   = client.get_latest_state()
            curr_hp = state["boss"]["hp"] if state else 0

            if curr_hp != stable_hp:
                stable_hp    = curr_hp
                stable_since = time.time()
            elif time.time() - stable_since >= BOSS_STABLE_SECS:
                break

        if stable_hp > 0:
            break

        print("[test] Boss HP 穩定在 0，繼續等待...")
        while time.time() < deadline:
            time.sleep(0.1)
            state = client.get_latest_state()
            if state and state.get("in_boss_room") and state["boss"]["hp"] > 0:
                break
    else:
        raise TimeoutError("Boss HP 穩定逾時")

    return state


def monitor_fight(client: HKBridgeClient) -> str:
    """
    監控戰鬥進行中，不送出任何動作。
    回傳 'win' / 'lose' / 'left_room'
    """
    prev_boss_hp   = None
    prev_player_hp = None

    while True:
        state = client.get_latest_state()
        if state is None:
            print("[test] 連線中斷")
            return "disconnect"

        boss_hp   = state["boss"]["hp"]
        player_hp = state["player"]["hp"]
        scene     = state["scene"]
        in_room   = state.get("in_boss_room", False)

        # 只在有變化時才印出
        if boss_hp != prev_boss_hp or player_hp != prev_player_hp:
            boss_pct   = state["boss"]["hp_pct"]   * 100
            player_pct = state["player"]["hp_pct"] * 100
            print(f"  場景={scene}  玩家HP={player_hp}({player_pct:.0f}%)  "
                  f"BossHP={boss_hp}({boss_pct:.0f}%)")
            prev_boss_hp   = boss_hp
            prev_player_hp = player_hp

        if boss_hp <= 0:
            return "win"
        if player_hp <= 0:
            return "lose"
        if not in_room:
            return "left_room"

        time.sleep(0.05)


def main():
    client = HKBridgeClient(HOST, PORT)

    print("=" * 55)
    print("  HKAIBridge 勝利重置測試")
    print("  請先手動走進 GG_Hornet_1")
    print("=" * 55)

    try:
        client.connect()
    except OSError as e:
        print(f"[test] 連線失敗：{e}")
        print("       請確認遊戲已開啟且 HKAIBridge Mod 已載入")
        return

    episode = 0
    while True:
        episode += 1
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  第 {episode} 局")
        print(sep)

        # 等待 Boss 房就緒
        try:
            state = wait_for_boss_room(client)
        except TimeoutError as e:
            print(f"[test] {e}，重試...")
            try:
                client.connect()
            except OSError:
                pass
            continue

        boss_name = state["boss"]["name"]
        boss_max  = state["boss"]["max_hp"]
        print(f"[test] Boss 就緒：{boss_name}  MaxHP={boss_max}")
        print(f"[test] 開始戰鬥！（自己操作，此腳本只監控）")

        # 監控戰鬥
        result = monitor_fight(client)

        # 判斷結果
        if result == "win":
            print(f"\n[test] ✓ 勝利！等待 Mod 重載 Boss 房...")
            t0 = time.time()

            # 等待場景切換（boss.hp 歸零後離開 boss room）
            while True:
                state = client.get_latest_state()
                if state is None:
                    break
                if not state.get("in_boss_room") or state["boss"]["hp"] <= 0:
                    # 等到下一局就緒（_wait_for_boss_room 會處理）
                    break
                time.sleep(0.1)

            print(f"[test] 等待 Boss 房重載（Mod 有 2 秒延遲）...")
            # 直接進入下一輪 wait_for_boss_room，讓它等到新的一局就緒
            elapsed = time.time() - t0
            print(f"[test] 本局結束，耗時 {elapsed:.1f}s")

        elif result == "lose":
            print(f"\n[test] ✗ 死亡，等待 Mod 重載 Boss 房...")
        elif result == "left_room":
            print(f"\n[test] 離開 Boss 房，等待重載...")
        else:
            print(f"\n[test] 異常結束（{result}）")

        # 回到頂端等下一局
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test] 手動中止")
