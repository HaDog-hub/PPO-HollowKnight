using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Modding;

namespace HKAIBridge.Network
{
    /// <summary>
    /// 輕量 TCP Server（Phase 1）
    ///
    /// 架構說明：
    ///   - 主執行緒（Unity）呼叫 Broadcast() 推送狀態
    ///   - 背景執行緒接受連線、讀取 Python 傳入的請求
    ///   - Phase 1 支援 get_state 指令（Python 主動查詢）
    ///   - Phase 2 起可在 _onCommand 擴充 reset / goto_scene 等指令
    ///
    /// 訊息格式：
    ///   每條訊息以 '\n' 結尾的 UTF-8 JSON 字串（newline-delimited JSON）
    /// </summary>
    public class TcpBridgeServer
    {
        private readonly int _port;
        private TcpListener? _listener;
        private readonly List<TcpClient> _clients = new List<TcpClient>();
        private Thread? _acceptThread;
        private volatile bool _running;

        public bool HasConnectedClients
        {
            get { lock (_clients) return _clients.Count > 0; }
        }

        // Phase 2 預留：接收到指令時的回呼
        public Action<string>? OnCommand;

        // Phase 1：供外部設置「當收到 get_state 時回傳當前狀態的 func」
        public Func<string>? GetStateFunc;

        public TcpBridgeServer(int port) => _port = port;

        public void Start()
        {
            _listener = new TcpListener(IPAddress.Loopback, _port);
            _listener.Start();
            _running = true;

            _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "HKAIBridge-Accept" };
            _acceptThread.Start();

            Logger.Log($"[HKAIBridge] TCP server listening on 127.0.0.1:{_port}");
        }

        // ── 接受連線 ────────────────────────────────────────────────────────────

        private void AcceptLoop()
        {
            while (_running)
            {
                try
                {
                    var client = _listener!.AcceptTcpClient();
                    client.NoDelay = true;  // 關閉 Nagle 演算法，降低延遲

                    lock (_clients) _clients.Add(client);
                    Logger.Log("[HKAIBridge] Python client connected");

                    // 新連線立刻推送一次當前狀態（type: "connected"）
                    if (GetStateFunc != null)
                        Send(client, GetStateFunc());

                    var t = new Thread(() => ReadLoop(client))
                    {
                        IsBackground = true,
                        Name         = "HKAIBridge-Read"
                    };
                    t.Start();
                }
                catch (Exception e) when (_running)
                {
                    Logger.LogError($"[HKAIBridge] Accept error: {e.Message}");
                }
            }
        }

        // ── 讀取 Python 傳入的指令 ───────────────────────────────────────────────

        private void ReadLoop(TcpClient client)
        {
            var stream = client.GetStream();
            var buf    = new byte[1024];
            var sb     = new StringBuilder();

            while (_running && client.Connected)
            {
                try
                {
                    int n = stream.Read(buf, 0, buf.Length);
                    if (n == 0) break;

                    sb.Append(Encoding.UTF8.GetString(buf, 0, n));

                    // 逐行解析指令（newline-delimited）
                    string raw;
                    while ((raw = ExtractLine(sb)) != null)
                    {
                        raw = raw.Trim();
                        if (raw.Length == 0) continue;

                        if (raw.Contains("\"get_state\"") && GetStateFunc != null)
                        {
                            // 直接在 Socket 執行緒回應，不需要主執行緒介入
                            Send(client, GetStateFunc());
                        }
                        else
                        {
                            // Phase 2：交給 CommandQueue 排入主執行緒執行
                            OnCommand?.Invoke(raw);
                        }
                    }
                }
                catch { break; }
            }

            lock (_clients) _clients.Remove(client);
            Logger.Log("[HKAIBridge] Python client disconnected");
        }

        // ── 廣播給所有已連線的 Python client ────────────────────────────────────

        public void Broadcast(string json)
        {
            var data = Encoding.UTF8.GetBytes(json + "\n");
            lock (_clients)
            {
                var dead = new List<TcpClient>();
                foreach (var c in _clients)
                {
                    try   { c.GetStream().Write(data, 0, data.Length); }
                    catch { dead.Add(c); }
                }
                foreach (var c in dead) _clients.Remove(c);
            }
        }

        // ── 工具方法 ─────────────────────────────────────────────────────────────

        private static void Send(TcpClient client, string json)
        {
            var data = Encoding.UTF8.GetBytes(json + "\n");
            try { client.GetStream().Write(data, 0, data.Length); }
            catch { }
        }

        /// <summary>從 StringBuilder 中提取第一行（含 '\n'），找不到回傳 null</summary>
        private static string? ExtractLine(StringBuilder sb)
        {
            var s = sb.ToString();
            int idx = s.IndexOf('\n');
            if (idx < 0) return null;
            sb.Remove(0, idx + 1);
            return s.Substring(0, idx);
        }

        public void Stop()
        {
            _running = false;
            _listener?.Stop();
        }
    }
}
