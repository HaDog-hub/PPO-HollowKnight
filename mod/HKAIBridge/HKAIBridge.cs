using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using HKAIBridge.Core;
using HKAIBridge.Network;
using HKAIBridge.State;
using Modding;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKAIBridge
{
    public class HKAIBridge : Mod, ITogglableMod
    {
        private PlayerTracker?   _player;
        private BossTracker?     _boss;
        private TcpBridgeServer? _server;

        // 記錄玩家第一次手動進入 Boss 房時遊戲用的完整參數
        // 之後自動重置時直接重播，保證出生點和相機與正常入場完全一致
        private GameManager.SceneLoadInfo? _lastBossEntryInfo;

        // 勝利重載進行中旗標：防止 OnSceneLoaded 在 win 路徑下重複啟動 LoadBossRoomDelayed
        private bool _winReloadInProgress = false;

        public override string GetVersion() => "0.1.0";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            _server = new TcpBridgeServer(11000);
            _server.GetStateFunc = () => Serialize(BuildSnapshot("connected"));
            _server.Start();

            _player = new PlayerTracker(() => Push("event"));
            _boss   = new BossTracker(type => Push(type), OnBossDied);

            // 攔截 BeginSceneTransition：記錄進入 Boss 房時的完整參數
            On.GameManager.BeginSceneTransition += OnBeginSceneTransition;

            // 攔截 EndBossScene：Python 連線時直接壓制，防止跳到 GG_Workshop
            // （重載 Boss 房改由 BossTracker 的 OnBossDied callback 處理）
            On.BossSceneController.EndBossScene += OnBossSceneControllerEndBossScene;

            UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;

            Log("HKAIBridge initialized — waiting for Python connection on port 11000");
        }

        // ── 勝利偵測：EndBossScene 壓制 + BossTracker callback ──────────────────

        private void OnBossSceneControllerEndBossScene(
            On.BossSceneController.orig_EndBossScene orig,
            BossSceneController self)
        {
            // Python 連線中：壓制 EndBossScene，不讓它跳到 GG_Workshop。
            // 【為什麼壓制？】
            // 若讓 EndBossScene 繼續執行，它會呼叫 BeginSceneTransition("GG_Workshop")，
            // 然後 OnSceneLoaded 又啟動 LoadBossRoomDelayed，與 OnBossDied 的
            // ReloadBossRoomAfterWin 產生雙重轉跳。
            bool hasClients = _server != null && _server.HasConnectedClients;
            if (hasClients && _lastBossEntryInfo != null)
            {
                Log($"[HKAIBridge] 壓制 EndBossScene（GG_Workshop 轉跳），由 BossTracker callback 處理重載");
                return;
            }
            orig(self);
        }

        // BossTracker 在 TakeDamage 偵測到 hp <= 0 時呼叫此 callback
        // 比 On.HealthManager.Die 更可靠：與廣播 boss.hp<=0 走同一條路徑
        private void OnBossDied()
        {
            bool hasClients = _server != null && _server.HasConnectedClients;
            if (!hasClients || _lastBossEntryInfo == null) return;

            Log($"[HKAIBridge] Boss 死亡（TakeDamage），2 秒後直接重載 Boss 房");
            _winReloadInProgress = true;
            GameManager.instance.StartCoroutine(ReloadBossRoomAfterWin());
        }

        private IEnumerator ReloadBossRoomAfterWin()
        {
            yield return new WaitForSeconds(2.0f);
            _winReloadInProgress = false;
            Log($"[HKAIBridge] 直接重載 Boss 房：{_lastBossEntryInfo!.SceneName}  gate={_lastBossEntryInfo.EntryGateName}");
            try
            {
                GameManager.instance.BeginSceneTransition(_lastBossEntryInfo);
            }
            catch (Exception e)
            {
                Log($"[HKAIBridge] 重載失敗：{e.Message}");
            }
        }

        // ── 記錄 Boss 房入口參數 ──────────────────────────────────────────────────

        private static readonly System.Collections.Generic.HashSet<string> GodhomeHubs
            = new System.Collections.Generic.HashSet<string>
            { "GG_Atrium", "GG_Workshop", "GG_Lobby", "GG_Spa" };

        private void OnBeginSceneTransition(
            On.GameManager.orig_BeginSceneTransition orig,
            GameManager self,
            GameManager.SceneLoadInfo info)
        {
            // 玩家手動進入 Boss 房時，記下這次的完整 SceneLoadInfo
            if (info.SceneName != null
                && info.SceneName.StartsWith("GG_")
                && !GodhomeHubs.Contains(info.SceneName))
            {
                _lastBossEntryInfo = info;
                Log($"[HKAIBridge] 記錄 Boss 房入口：{info.SceneName}  gate={info.EntryGateName}");
            }
            orig(self, info);
        }

        // ── 自動重置 ─────────────────────────────────────────────────────────────

        private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
        {
            bool hasClients = _server != null && _server.HasConnectedClients;
            Log($"[HKAIBridge] OnSceneLoaded: scene={scene.name}, hasClients={hasClients}");

            // 勝利路徑已由 ReloadBossRoomAfterWin 處理，跳過死亡路徑的 hub 偵測
            if (_winReloadInProgress)
            {
                Log($"[HKAIBridge] Win reload 進行中，跳過 hub 偵測");
                return;
            }

            // 死亡路徑：偵測到 Godhome hub（GG_Workshop/GG_Atrium 等）→ 重載 Boss 房
            string? target      = _boss?.LastBossScene;
            bool    isGodhomeHub = scene.name.StartsWith("GG_") && scene.name != target;
            if (isGodhomeHub && hasClients && target != null)
            {
                Log($"[HKAIBridge] 偵測到 Godhome hub（{scene.name}），2 秒後載入：{target}");
                GameManager.instance.StartCoroutine(LoadBossRoomDelayed(target));
            }
        }

        private IEnumerator LoadBossRoomDelayed(string target)
        {
            yield return new WaitForSeconds(2.0f);
            Log($"[HKAIBridge] 載入 Boss 房：{target}");

            try
            {
                if (_lastBossEntryInfo != null && _lastBossEntryInfo.SceneName == target)
                {
                    // 重播第一次手動入場的完整參數，出生點和相機與正常入場完全一致
                    Log($"[HKAIBridge] 重播入口參數  gate={_lastBossEntryInfo.EntryGateName}");
                    GameManager.instance.BeginSceneTransition(_lastBossEntryInfo);
                }
                else
                {
                    // 還沒有記錄（理論上不會發生，但當保底用）
                    Log("[HKAIBridge] 尚無入口記錄，使用 LoadScene");
                    GameManager.instance.LoadScene(target);
                }
            }
            catch (Exception e)
            {
                Log($"[HKAIBridge] 載入失敗：{e.Message}");
            }
        }

        // ── 狀態推送 ─────────────────────────────────────────────────────────────

        private void Push(string type)
        {
            if (_server == null) return;
            _server.Broadcast(Serialize(BuildSnapshot(type)));
        }

        private GameStateSnapshot BuildSnapshot(string type)
        {
            string scene      = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            bool   inBossRoom = _boss != null && _boss.IsActive;

            var snap = new GameStateSnapshot
            {
                type       = type,
                timestamp  = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                scene      = scene,
                inBossRoom = inBossRoom,
            };

            int playerHp    = _player?.CurrentHp ?? 0;
            int playerMaxHp = _player?.MaxHp     ?? 0;
            snap.player.hp    = playerHp;
            snap.player.maxHp = playerMaxHp;
            snap.player.hpPct = playerMaxHp > 0 ? (float)playerHp / playerMaxHp : 0f;

            int bossHp    = _boss?.CurrentHp ?? 0;
            int bossMaxHp = _boss?.MaxHp     ?? 0;
            snap.boss.name  = _boss?.BossName ?? "none";
            snap.boss.hp    = bossHp;
            snap.boss.maxHp = bossMaxHp;
            snap.boss.hpPct = bossMaxHp > 0 ? (float)bossHp / bossMaxHp : 0f;

            return snap;
        }

        // ── 手寫 JSON 序列化 ──────────────────────────────────────────────────────

        private static string Serialize(GameStateSnapshot s)
        {
            string F(float v) => v.ToString("F4", CultureInfo.InvariantCulture);
            string B(bool v)  => v ? "true" : "false";

            return "{"
                + $"\"type\":\"{s.type}\","
                + $"\"timestamp\":{s.timestamp},"
                + $"\"scene\":\"{Escape(s.scene)}\","
                + $"\"in_boss_room\":{B(s.inBossRoom)},"
                + "\"player\":{"
                +   $"\"hp\":{s.player.hp},"
                +   $"\"max_hp\":{s.player.maxHp},"
                +   $"\"hp_pct\":{F(s.player.hpPct)}"
                + "},"
                + "\"boss\":{"
                +   $"\"name\":\"{Escape(s.boss.name)}\","
                +   $"\"hp\":{s.boss.hp},"
                +   $"\"max_hp\":{s.boss.maxHp},"
                +   $"\"hp_pct\":{F(s.boss.hpPct)}"
                + "}"
                + "}";
        }

        private static string Escape(string s) =>
            s.Replace("\\", "\\\\").Replace("\"", "\\\"");

        // ── 卸載 ─────────────────────────────────────────────────────────────────

        public void Unload()
        {
            On.GameManager.BeginSceneTransition -= OnBeginSceneTransition;
            On.BossSceneController.EndBossScene -= OnBossSceneControllerEndBossScene;
            UnityEngine.SceneManagement.SceneManager.sceneLoaded -= OnSceneLoaded;
            _player?.Unload();
            _boss?.Unload();
            _server?.Stop();
        }
    }
}
