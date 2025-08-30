# bridge_to_agent.py —— 本地RAG → Coze 智能体桥接（可被他人访问版）
# 作用：把“用户问题 + 本地RAG命中片段（证据区）”发送到 Coze Bot，拿回最终中文答案
# 仅依赖：requests、fastapi、pydantic（以及你本机正在跑的 app.py:8000）

import os
import json
import time
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from fastapi import Header

# ===================== 配置区 =====================
# 【重点】你的本地 RAG 服务地址
# - 如果只在本机自测，可改回: http://127.0.0.1:8000/ask_debug
# - 如果要让“同一局域网里”的同事访问到你的Bridge，并让Bridge去请求你这台机的RAG，就要写成你的电脑的局域网IP
LOCAL_RAG_URL = os.getenv("LOCAL_RAG_URL", "http://127.0.0.1:8000/ask_debug")

# Coze API 配置（务必先在环境变量里配置你自己的 Token 与 BotID）
# 国内站： https://api.coze.cn/open_api/v2
# 海外站： https://api.coze.com/open_api/v2
COZE_BASE      = os.getenv("COZE_BASE", "https://api.coze.cn/open_api/v2")
COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "")          # 必填：你的 Coze API Key
COZE_BOT_ID    = os.getenv("COZE_BOT_ID", "")             # 必填：你的 Bot ID
COZE_USER_ID   = os.getenv("COZE_USER_ID", "lan_user")    # 可随意指定一个“用户ID”

# ===================== 请求体模型 =====================
class BridgeReq(BaseModel):
    question: str
    topk: int = 4
    mode: str = "answer"   # "answer"：RAG+Coze；"check"：只看RAG命中与证据，不发Coze

# ===================== 工具函数 =====================
def call_local_rag(question: str, topk: int = 4) -> dict:
    """
    调用你本地的 RAG 接口，拿命中片段。
    建议配合 app.py 的 /ask_debug 使用：返回 {"hits":[{score, source, idx, text}, ...]}
    """
    try:
        r = requests.post(LOCAL_RAG_URL, json={"question": question, "topk": topk}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"RAG调用失败: {e}", "results": [], "hits": []}

def build_context_from_hits(hits: list[dict], max_refs: int = 3, max_each: int = 300) -> str:
    """
    把命中片段拼成【证据区】字符串。最多取 max_refs 条，每条最多 max_each 字符。
    兼容 /ask_debug 的字段(text) 和 /ask 的字段(snippet)。
    """
    if not hits:
        return ""
    parts = []
    for i, h in enumerate(hits[:max_refs], 1):
        text = (h.get("text") or h.get("snippet") or "").replace("\n", " ")
        if len(text) > max_each:
            text = text[:max_each] + "…"
        src = h.get("source") or "unknown"
        idx = h.get("idx") or h.get("paragraphIndex") or "?"
        parts.append(f"[{i}] {src}#段{idx}: {text}")
    return "\n".join(parts)

def call_coze_chat(question: str, context: str) -> dict:
    """
    把“用户问题 + 证据区”发到 Coze，并“强制”抽取最后一条 assistant 文本作为 final。
    若失败/报错，会把错误信息作为 final 返回，便于你直接看到问题。
    """
    if not COZE_API_TOKEN or not COZE_BOT_ID:
        return {"ok": False, "status": 0, "final": "（缺少 COZE_API_TOKEN / COZE_BOT_ID 环境变量）"}

    prompt = f"""你是PLUS生活服务包客服助手。请严格依据【证据区】回答，禁止编造未在证据中的信息。
- 先给结论（1-3条要点），再给依据编号（如 [1][3]）
- 术语统一：运费券=免费寄件；开通=开卡；续约=续费
- 若证据不足或冲突，请输出：“需要人工复核：原因…”，不要给武断结论

【用户问题】
{question}

【证据区】
{context}
"""

    url = f"{COZE_BASE}/chat"
    headers = {
        "Authorization": f"Bearer {COZE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "bot_id": COZE_BOT_ID,
        "user": f"{COZE_USER_ID}-{int(time.time() * 1000) % 1000000}",  # 每次一个新会话，如果你确实需要连续多轮，再把这行改回固定的 COZE_USER_ID
        "query": prompt,
        "stream": False
    }

    def _pick_last_assistant(data: dict) -> str | None:
        """
        选择规则（按优先级）：
        1) 第一优先：messages 里第一条 role=assistant 且 type in {"answer","final","reply"} 的 content/text
        2) 第二优先：messages 里 role=assistant 但 type 不是 {"follow_up","verbose"} 的“最后一条有文本”
        3) 第三优先：messages 里“最后一条有文本”的（任何 type）
        4) 兜底：data/data 顶层的 content/answer
        """
        if not isinstance(data, dict):
            return None

        # 错误包优先直接返回错误信息
        code = data.get("code")
        if isinstance(code, int) and code not in (0, 200):
            return data.get("msg") or data.get("message") or json.dumps(data, ensure_ascii=False)

        d = data.get("data") if isinstance(data.get("data"), dict) else data

        # 统一拿列表
        msgs = None
        for key in ("messages", "list", "message_list"):
            if isinstance(d.get(key), list):
                msgs = d.get(key)
                break
        if not isinstance(msgs, list) or not msgs:
            # 没有 messages，看看是否直接给了 content/answer
            direct = d.get("content") or d.get("answer") or data.get("content") or data.get("answer")
            return (direct.strip() if isinstance(direct, str) and direct.strip() else None)

        # 1) 第一优先：第一条 assistant 且 type=answer/final/reply
        for m in msgs:
            role = (m.get("role") or m.get("sender") or m.get("sender_type") or "").lower()
            mtype = (m.get("type") or "").lower()
            if role == "assistant" and mtype in ("answer", "final", "reply"):
                t = m.get("content") or m.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()

        # 2) 第二优先：从后往前找 assistant，且类型不是 follow_up/verbose 的“最后一条”
        for m in reversed(msgs):
            role = (m.get("role") or m.get("sender") or m.get("sender_type") or "").lower()
            mtype = (m.get("type") or "").lower()
            if role == "assistant" and mtype not in ("follow_up", "verbose"):
                t = m.get("content") or m.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()

        # 3) 第三优先：就拿最后一条“有文本”的
        for m in reversed(msgs):
            t = m.get("content") or m.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()

        # 4) 顶层兜底
        top = d.get("content") or d.get("answer") or data.get("content") or data.get("answer")
        return (top.strip() if isinstance(top, str) and top.strip() else None)

        return None

    try:
        r = requests.post(url, headers=headers, json=body, timeout=45)
        status = r.status_code
        text = r.text
        try:
            data = r.json()
        except Exception:
            data = None

        final = _pick_last_assistant(data) if isinstance(data, dict) else None
        if not final:
            final = text.strip() if isinstance(text, str) and text.strip() else "（未提取到回答，且无可读返回）"

        return {
            "ok": (200 <= status < 300),
            "status": status,
            "final": final,
            "data": data,
            "raw": text[:2000],
            "via": "force_pick_last_assistant"
        }
    except Exception as e:
        return {"ok": False, "status": 0, "final": f"（请求异常：{repr(e)}）"}

def ask_pipeline(question: str, topk: int = 4, mode: str = "answer") -> dict:
    """
    主流程：
      - mode="check": 只返回 RAG 命中与证据（不调用 Coze）
      - mode="answer": RAG→拼证据→调用 Coze→返回最终答案
    """
    rag = call_local_rag(question, topk=topk)
    hits = rag.get("results") or rag.get("hits") or rag.get("citations") or []
    # —— 新增：二次重排 ——
    hits = reweight_hits(question, hits)
    # —— 用重排后的命中构建证据区（取前3条）——
    context = build_context_from_hits(hits, max_refs=3)

    if not context.strip():
        return {
            "stage": "answer",
            "question": question,
            "context": context,
            "coze_result": {
                "ok": True,
                "status": 200,
                "messages": [{"role": "assistant", "type": "answer",
                              "content": "需要人工复核：知识库未命中或证据不足。"}],
                "final": "需要人工复核：知识库未命中或证据不足。"
            }
        }

    if mode != "answer":
        return {
            "stage": "check_only",
            "question": question,
            "hits_count": len(hits),
            "context": context,
            "raw_hits": hits[:4],
        }

    coze = call_coze_chat(question, context)
    return {
        "stage": "answer",
        "question": question,
        "context": context,
        "coze_result": coze
    }

# ===================== FastAPI 应用与路由 =====================
app = FastAPI(title="Bridge to Coze Bot", version="1.0.0")

# 统一把 JSON 响应头加上 charset=utf-8，避免中文显示成乱码
@app.middleware("http")
async def _force_utf8_json(request, call_next):
    resp = await call_next(request)
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json") and "charset=" not in ctype:
        resp.headers["content-type"] = "application/json; charset=utf-8"
    return resp

@app.on_event("startup")
async def _on_start():
    print("[bridge] STARTED:", __file__)

@app.exception_handler(Exception)
async def _global_ex_handler(request, exc):
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("\n[GLOBAL][EXCEPTION]", tb, "\n")
    return JSONResponse(
        status_code=200,
        content={"ok": False, "where": "global", "error": str(exc), "traceback": tb[:2000]},
    )

@app.get("/whoami")
def whoami():
    import os
    return {"app": "bridge_coze", "pid": os.getpid(), "port_hint": 8016, "rag_url": LOCAL_RAG_URL}

# UTF-8 自检（排查中文乱码）
@app.get("/utf8-test/json")
def utf8_test_json():
    return JSONResponse({"msg": "中文OK，JSON没问题"}, media_type="application/json; charset=utf-8")

@app.get("/utf8-test/text")
def utf8_test_text():
    return PlainTextResponse("中文OK，纯文本没问题")

# 运行状况检查
@app.get("/health")
def health():
    rag_ok = False
    try:
        r = requests.post(LOCAL_RAG_URL, json={"question":"ping","topk":1}, timeout=5)
        rag_ok = (r.status_code == 200)
    except Exception:
        rag_ok = False
    return {
        "ok": True,
        "rag_url": LOCAL_RAG_URL,
        "rag_ok": rag_ok,
        "coze_base": COZE_BASE,
        "coze_token_set": bool(COZE_API_TOKEN),
        "coze_bot_set": bool(COZE_BOT_ID),
    }

# 主入口（JSON）：返回 context + coze_result
@app.post("/bridge/ask")
def bridge_ask(req: BridgeReq):
    q = (req.question or "").strip()
    if not q:
        return {"ok": False, "error": "缺少 question"}
    topk = int(req.topk)
    mode = (req.mode or "answer").lower()
    out = ask_pipeline(q, topk=topk, mode=mode)
    return {"ok": True, **out}

# 一把梭（纯文本）：最适合在平台里直接接收最终答案
@app.post("/bridge/ask-and-wait")
def bridge_ask_and_wait(req: BridgeReq, x_bridge_secret: str | None = Header(None)):
    secret = os.getenv("BRIDGE_SECRET", "")
    if secret and x_bridge_secret != secret:
        return PlainTextResponse("Unauthorized", status_code=401)
    q = (req.question or "").strip()
    if not q:
        return PlainTextResponse("（缺少 question）", status_code=200)
    topk = int(req.topk)
    rag = call_local_rag(q, topk=topk)
    hits = rag.get("results") or rag.get("hits") or rag.get("citations") or []
    context = build_context_from_hits(hits, max_refs=3)
    coze = call_coze_chat(q, context)
    # 这里改一下：
    final = (coze.get("final") or "").strip() or "（抱歉，未拿到答案）"
    return PlainTextResponse(final)   # 直接返回纯文本

from fastapi import Request

def _check_secret(req: Request) -> bool:
    secret = os.getenv("BRIDGE_SECRET", "")
    if not secret:
        return True  # 未设置则不校验（本地开发用），线上务必设置
    return req.headers.get("X-Bridge-Secret") == secret

@app.post("/debug/rag-only")
async def debug_rag_only(req: BridgeReq, request: Request):
    if not _check_secret(request):
        return PlainTextResponse("Unauthorized", status_code=401)
    rag = call_local_rag(req.question, topk=req.topk)
    hits = rag.get("results") or rag.get("hits") or rag.get("citations") or []
    context = build_context_from_hits(hits, max_refs=4, max_each=200)
    return {"question": req.question, "hits_count": len(hits), "context": context, "raw_hits": hits[:4]}

@app.post("/debug/coze-raw")
async def debug_coze_raw(req: BridgeReq, request: Request):
    if not _check_secret(request):
        return PlainTextResponse("Unauthorized", status_code=401)
    rag = call_local_rag(req.question, topk=req.topk)
    hits = rag.get("results") or rag.get("hits") or rag.get("citations") or []
    context = build_context_from_hits(hits, max_refs=3)
    coze = call_coze_chat(req.question, context)
    return {"question": req.question, "context": context, "status": coze.get("status"),
            "final_picked": coze.get("final"), "data": coze.get("data"), "raw": coze.get("raw")}

# 诊断信息
@app.get("/diag")
def diag():
    import os, time
    return {
        "signature": "bridge_diag_v1",
        "file": __file__,
        "pid": os.getpid(),
        "time": int(time.time())
    }

# 直接作为脚本跑一把（可选）
if __name__ == "__main__":
    demo = ask_pipeline("为什么积分只有9分", topk=3, mode="check")

    print(json.dumps(demo, ensure_ascii=False, indent=2))

