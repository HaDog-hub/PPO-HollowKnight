namespace HKAIBridge.State
{
    /// <summary>
    /// 玩家（小騎士）當前狀態
    /// </summary>
    public class PlayerState
    {
        public int hp;
        public int maxHp;
        public float hpPct;  // 0.0 ~ 1.0，供 AI 泛化使用
    }

    /// <summary>
    /// Boss 當前狀態
    /// </summary>
    public class BossState
    {
        public string name = "none";
        public int hp;
        public int maxHp;
        public float hpPct;  // 0.0 ~ 1.0，供 AI 泛化使用
    }

    /// <summary>
    /// 推送的完整快照
    /// type:
    ///   "event"       — HP 變化（主動推送）
    ///   "response"    — 回應 get_state 查詢
    ///   "scene_enter" — 進入或離開 Boss 房
    ///   "connected"   — Python 剛連線時的初始狀態
    /// </summary>
    public class GameStateSnapshot
    {
        public string type = "event";
        public long timestamp;
        public string scene = "";
        public bool inBossRoom;      // 是否在 GG_ Boss 房內
        public PlayerState player = new PlayerState();
        public BossState boss = new BossState();
    }
}
