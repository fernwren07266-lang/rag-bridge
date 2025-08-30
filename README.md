# RAG → Coze 智能体桥接（本地可运行 Demo）

> **目标**：在 **Coze 智能体** 上输入问题，后端调用我本地的 **RAG 服务**（BM25 + 语义重排），把“命中证据 + 最终答案”返回给 Coze。  
这样坐席/运营能同时看到答案和证据，还能通过调参优化检索效果，让回答更稳定。

---

## ✨ 项目亮点
- **RAG 检索可控**：BM25 + 可选中文向量（BAAI/bge-small-zh-v1.5）重排
- **切分更稳**：优先按“编号 / 空行 / Q&A”切段，避免把完整规则或 Q&A 切断
- **桥接 Coze**：通过 `/bridge/ask-and-wait` 接口，把“用户问题 + 命中证据”发给 Coze Bot
- **可观测 / 可调参**：提供 `/debug/rag-only` 等接口，方便快速调试和参数优化

---

## 🗂 目录结构
.
├─ app.py                # 本地 RAG 服务（/ask, /ask_debug, /kb/search 等）
├─ bridge_to_agent.py    # 桥接到 Coze（/bridge/ask, /bridge/ask-and-wait 等）
├─ rag_step1_bm25.py     # 检索器与规则文档切分（编号/空行/Q&A 友好）
├─ kb/
│  └─ demo_rules.txt     # 演示用知识库
├─ .gitignore            # 忽略 .venv、__pycache__、*.env 等
├─ LICENSE               # MIT License
└─ README.md             # 项目说明文档

---

## ⚙️ 配置环境变量

请先复制示例文件并修改：

```bash
cp .env.example .env
```

---

## 🚀 本地运行
> 需 Python 3.10+，建议用虚拟环境

1. **安装依赖**
   ```bash
   pip install fastapi uvicorn requests pydantic==1.* rank-bm25 jieba sentence-transformers numpy
   ```

2. **启动 RAG 服务**（端口 8000）
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   健康检查：
   - 浏览器打开：`http://127.0.0.1:8000/health`
   - 预览知识块：`http://127.0.0.1:8000/kb/chunks`

3. **启动 Bridge（桥接到 Coze）**（端口 8016）
   ```bash
   # 设置必要环境变量
   set COZE_BASE=https://api.coze.cn/open_api/v2
   set COZE_API_TOKEN=替换为你的CozeToken
   set COZE_BOT_ID=替换为你的BotID
   set COZE_USER_ID=demo_user
   set BRIDGE_SECRET=设置一个随意的密钥

   uvicorn bridge_to_agent:app --host 0.0.0.0 --port 8016 --reload
   ```

---

## 🔌 常用接口（便于调参与联调）
- **只看命中（不走 Coze）**
  ```
  POST http://127.0.0.1:8016/debug/rag-only
  {
    "question": "洗车多久过期",
    "topk": 4
  }
  ```

- **拿最终答案（走 Coze）**
  ```
  POST http://127.0.0.1:8016/bridge/ask-and-wait
  {
    "question": "积分兑换的商品是否可以开发票",
    "topk": 4,
    "mode": "answer"
  }
  ```
  > ⚠️ 建议请求头里加：`X-Bridge-Secret: abc123`

---

## 🌉 和 Coze 对接
在 Coze 工作流的 **HTTP 请求节点**里，调用：
```
POST https://你的域名.ngrok-free.app/bridge/ask-and-wait
```

请求体：
```json
{ "question": "${input}", "topk": 4, "mode": "answer" }
```

如果同时需要“命中调试”，可以用并行节点，把两个结果都传入一个 **代码节点**，最后只输出：
```json
{
  "answerfinal": "（最终答案纯文本）",
  "hitscontext": "（命中证据纯文本，可展示给坐席）"
}
```

---

## 🧪 PowerShell 示例
```powershell
$body = @{ question = "洗车多久过期"; topk = 4; mode = "answer" } | ConvertTo-Json -Compress
$headers = @{ "Content-Type" = "application/json; charset=utf-8"; "X-Bridge-Secret" = "abc123" }
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8016/bridge/ask-and-wait" -Headers $headers -Body $body
```

---

## 🔐 安全与合规
- **不要上传** 公司真实规则文档、API Token、Cookie 等敏感信息
- `.gitignore` 已配置忽略 `.env`、本地 KB 文件等
- 如果不小心提交了密钥，请**立刻改密钥**并清理仓库历史

---

## ❓常见问题
- **下载模型失败**：已在 `rag_step1_bm25.py` 中支持 HuggingFace 镜像 `HF_ENDPOINT`
- **报 401 Unauthorized**：Bridge 启用了 `X-Bridge-Secret`，请求时需带一致的值
- **回答格式混乱**：在代码节点解析 JSON，只把 `answerfinal` 输出给 LLM

---

## 📄 许可证
MIT License — 可自由使用、修改、二次开发，但请保留版权声明
