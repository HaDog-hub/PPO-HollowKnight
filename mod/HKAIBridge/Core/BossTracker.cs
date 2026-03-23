using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKAIBridge.Core
{
    /// <summary>
    /// 追蹤尋神者模式（GG_ 場景）中 Boss 的血量變化
    ///
    /// 推送時機：
    ///   1. 進入 GG_ 場景時（scene_enter，boss 欄位為 none）
    ///   2. Boss HealthManager 首次偵測到時（scene_enter，含完整 boss 資料）
    ///   3. Boss HP 變化時（event）
    ///   4. 離開 GG_ 場景時（scene_enter，inBossRoom=false）
    /// </summary>
    public class BossTracker
    {
        // HP 變化回呼（帶 type 參數）
        private readonly Action<string> _onChanged;
        // Boss 死亡回呼（hp 首次降到 <= 0 時觸發一次）
        private readonly Action? _onBossDied;

        private HealthManager? _boss;
        private int _cachedMaxHp;
        private int _lastHp;
        private bool _bossDeathFired;   // 確保同一局只觸發一次死亡回呼

        private const int MinBossHp = 100;

        public bool          IsActive      => _boss != null;
        public bool          IsInBossRoom  => UnityEngine.SceneManagement.SceneManager
                                              .GetActiveScene().name.StartsWith("GG_");
        public int           CurrentHp    => _boss?.hp ?? 0;
        public int           MaxHp        => _cachedMaxHp;
        public string        BossName     => _boss?.gameObject?.name ?? "none";
        public string?       LastBossScene { get; private set; }
        public HealthManager? TrackedBossHm => _boss;   // 供 HKAIBridge 判斷是否為追蹤的 Boss

        public BossTracker(Action<string> onChanged, Action? onBossDied = null)
        {
            _onChanged  = onChanged;
            _onBossDied = onBossDied;
            UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;
            On.HealthManager.Start      += OnHealthManagerStart;
            On.HealthManager.TakeDamage += OnHealthManagerTakeDamage;
        }

        // 場景切換：重置狀態並推送 scene_enter 事件
        private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            _boss           = null;
            _cachedMaxHp    = 0;
            _lastHp         = 0;
            _bossDeathFired = false;

            // 不論進入或離開 Boss 房都推送，讓 Python 知道場景已切換
            _onChanged("scene_enter");
        }

        // Boss HealthManager 啟動時：首次偵測並推送
        private void OnHealthManagerStart(On.HealthManager.orig_Start orig, HealthManager self)
        {
            orig(self);

            string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            if (!scene.StartsWith("GG_")) return;
            if (self.hp < MinBossHp)      return;

            if (_boss == null || self.hp > _cachedMaxHp)
            {
                _boss         = self;
                _cachedMaxHp  = self.hp;
                _lastHp       = self.hp;
                LastBossScene = scene;   // 記下這個 Boss 所在的場景，供 auto-reset 使用
                Modding.Logger.Log($"[HKAIBridge] Boss detected: {self.gameObject.name} HP={self.hp} scene={scene}");

                // Boss 就緒，再推一次 scene_enter（此時含完整 boss 資料）
                _onChanged("scene_enter");
            }
        }

        // Boss 被打時偵測 HP 變化
        private void OnHealthManagerTakeDamage(
            On.HealthManager.orig_TakeDamage orig,
            HealthManager self,
            HitInstance hitInstance)
        {
            orig(self, hitInstance);

            if (self != _boss)      return;
            if (self.hp == _lastHp) return;

            _lastHp = self.hp;
            _onChanged("event");

            // Boss 死亡偵測：HP 首次降到 <= 0，觸發一次死亡回呼
            if (self.hp <= 0 && !_bossDeathFired)
            {
                _bossDeathFired = true;
                _onBossDied?.Invoke();
            }
        }

        public void Unload()
        {
            UnityEngine.SceneManagement.SceneManager.sceneLoaded -= OnSceneLoaded;
            On.HealthManager.Start      -= OnHealthManagerStart;
            On.HealthManager.TakeDamage -= OnHealthManagerTakeDamage;
        }
    }
}
