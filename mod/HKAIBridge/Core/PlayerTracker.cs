using System;
using Modding;

namespace HKAIBridge.Core
{
    /// <summary>
    /// 追蹤小騎士的血量變化
    ///
    /// 解法：
    ///   HeroUpdateHook 每幀比較 PlayerData.instance.health 與快取值，
    ///   有差異即推送。最多 1 幀延遲（≈16ms），不依賴任何 hook 時機。
    ///   回血另掛 On.PlayerData.AddHealth 即時捕捉。
    /// </summary>
    public class PlayerTracker
    {
        private readonly Action _onChanged;

        private int _cachedHp    = -1;
        private int _cachedMaxHp = -1;

        public int CurrentHp => _cachedHp    >= 0 ? _cachedHp    : (PlayerData.instance?.health    ?? 0);
        public int MaxHp     => _cachedMaxHp >= 0 ? _cachedMaxHp : (PlayerData.instance?.maxHealth ?? 0);

        public PlayerTracker(Action onChanged)
        {
            _onChanged = onChanged;
            ModHooks.HeroUpdateHook      += OnHeroUpdate;
            On.PlayerData.AddHealth      += OnAddHealth;
        }

        // 每幀輪詢：只要 HP 有變化立即推送（含扣血 / 死亡歸零）
        private void OnHeroUpdate()
        {
            int hp    = PlayerData.instance?.health    ?? 0;
            int maxHp = PlayerData.instance?.maxHealth ?? 0;

            // maxHp > 0 過濾場景載入時的初始零值
            if (maxHp > 0 && (hp != _cachedHp || maxHp != _cachedMaxHp))
                SetCache(hp, maxHp);
        }

        // 回血（撿心、bench 休息等）—— AddHealth 後立即捕捉，不等下一幀
        private void OnAddHealth(On.PlayerData.orig_AddHealth orig, PlayerData self, int amount)
        {
            orig(self, amount);
            SetCache(self.health, self.maxHealth);
        }

        private void SetCache(int hp, int maxHp)
        {
            if (hp == _cachedHp && maxHp == _cachedMaxHp) return;
            _cachedHp    = hp;
            _cachedMaxHp = maxHp;
            _onChanged();
        }

        public void Unload()
        {
            ModHooks.HeroUpdateHook      -= OnHeroUpdate;
            On.PlayerData.AddHealth      -= OnAddHealth;
        }
    }
}
