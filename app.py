# app.py
import os, time
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.responses import JSONResponse
from fastapi import Query
from pathlib import Path
from rag_step1_bm25 import (
    KB_DIR, clean_text, debug_split_paragraphs_from_text,
    debug_pack_paragraphs_to_blocks
)

# 从你的检索脚本里导入
from rag_step1_bm25 import get_retriever, USE_SEMANTIC

# ====== 启动时加载检索器 ======
retriever = get_retriever()

app = FastAPI(title="JD PLUS RAG Service")

@app.middleware("http")
async def _force_utf8_json(request, call_next):
    resp = await call_next(request)
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json") and "charset=" not in ctype:
        resp.headers["content-type"] = "application/json; charset=utf-8"
    return resp

# ===（可选）配置你们公司内网大模型 API 地址和密钥（没有就先空着，会自动降级）===
INTERNAL_LLM_URL   = os.getenv("INTERNAL_LLM_URL", "")   # 例: http://llm-internal/api/chat/completions
INTERNAL_LLM_TOKEN = os.getenv("INTERNAL_LLM_TOKEN", "") # 例: xxx

class AskReq(BaseModel):
    question: str
    topk: int = 4
    session_id: str | None = None
    meta: dict | None = None

def build_prompt(question: str, hits: list[dict]) -> str:
    """把命中的片段拼成【证据区】提示词，压住瞎编"""
    refs = []
    for i, h in enumerate(hits, 1):
        # 片段不要太长，避免 token 爆炸
        snippet = h["text"].replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        refs.append(f"[{i}] {h['source']}#段{h['idx']}: {snippet}")
    context = "\n".join(refs)
    prompt = f"""你是京东PLUS生活服务包客服助手。请仅依据【证据区】回答，禁止编造未在证据中的信息。
- 先给结论（1-3条要点，简洁），再给依据编号（如 [1][3]）
- 术语统一：运费券=免费寄件；开通=开卡；续约=续费
- 若证据不足或冲突，请输出：“需要人工复核：原因…”，不要给武断结论

【用户问题】
{question}

【证据区】
{context}
"""
    return prompt

def call_llm_or_rule(prompt: str, hits: list[dict]) -> str:
    """优先调用公司内网大模型；没配置就用规则式兜底"""
    if INTERNAL_LLM_URL and INTERNAL_LLM_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {INTERNAL_LLM_TOKEN}"}
            payload = {"model": "internal-default",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.2, "max_tokens": 512}
            r = requests.post(INTERNAL_LLM_URL, json=payload, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            return (data.get("choices", [{}])[0]
                        .get("message", {}).get("content", "")).strip() or "（模型无响应）"
        except Exception as e:
            print("[warn] 调用内网LLM失败，自动用规则兜底：", e)

    # —— 规则兜底（没有LLM时也能跑通）——
    if not hits:
        return "需要人工复核：知识库未命中。"
    first = hits[0]["text"].replace("\n", " ")
    if len(first) > 220: first = first[:220] + "…"
    return f"结论：请参考规则片段[1]。\n依据：[1] {first}"

def calc_confidence(hits: list[dict]) -> float:
    """简单置信度：Top1分 + 与Top2差值。后续你可以再调。"""
    if not hits: return 0.0
    s1 = hits[0]["score"]
    s2 = hits[1]["score"] if len(hits) > 1 else 0.0
    conf = min(0.95, 0.55 + 0.05*s1 + 0.15*max(0, s1 - s2))
    return round(conf, 2)

@app.get("/whoami")
def whoami():
    import os
    return {
        "app": "rag",
        "from_file": __file__,
        "pid": os.getpid(),
        "port_hint": 8000
    }


@app.get("/health")
def health():
    return {"ok": True, "use_semantic": bool(USE_SEMANTIC)}

@app.post("/reload")
def reload_kb():
    """当你更新了 kb/ 文件后，调用这个接口热加载"""
    global retriever
    retriever = get_retriever()
    return {"ok": True, "chunks": len(retriever.chunks)}

import time

# ① 把命中片段拼成“证据区”+简单回答（先结论后依据）
def build_simple_answer(question: str, hits: list[dict]) -> tuple[str, list[dict]]:
    if not hits:
        return "需要人工复核：知识库未命中。", []

    # 证据片段（截断，防止太长）
    citations = []
    lines = []
    for i, h in enumerate(hits, 1):
        snip = h["text"].replace("\n", " ")
        if len(snip) > 160:
            snip = snip[:160] + "…"
        citations.append({"source": h["source"], "idx": h["idx"], "snippet": snip})
        lines.append(f"[{i}] {h['source']}#段{h['idx']}: {snip}")

    # 简单规则：用 Top1 片段先给结论，再给依据编号
    answer = (
        "结论：请参考以下规则要点整理的结论，具体以引用片段为准。\n"
        f"依据：见 {''.join(f'[{i+1}]' for i in range(len(hits)))}"
    )
    return answer, citations

# ② 粗略置信度（Top1分+与Top2差值）
def calc_confidence(hits: list[dict]) -> float:
    if not hits:
        return 0.0
    s1 = hits[0]["score"]
    s2 = hits[1]["score"] if len(hits) > 1 else 0.0
    conf = min(0.95, 0.55 + 0.05*s1 + 0.15*max(0, s1 - s2))
    return round(conf, 2)

# ③ 把“检索→答案→结构化返回”封装
def make_response(question: str, hits: list[dict]) -> dict:
    t0 = time.time()
    answer, citations = build_simple_answer(question, hits)
    resp = {
        "answer": answer,
        "citations": citations,
        "confidence": calc_confidence(hits),
        "latency_ms": int((time.time() - t0) * 1000),
        "fallback": (answer.startswith("需要人工复核"))
    }
    return resp

@app.post("/ask")
def ask(req: AskReq):
    hits = retriever.retrieve(req.question, topk=req.topk)
    return make_response(req.question, hits)

# === 调试用：查看已切好的知识库片段 ===
@app.get("/kb/chunks")
def kb_chunks(limit: int = 10, show_chars: int = 120):
    """
    返回前 N 条切好的片段预览，便于确认 RAG 是否加载成功。
    - limit: 返回多少条
    - show_chars: 每条展示多少字符
    """
    try:
        chunks = getattr(retriever, "chunks", [])
        preview = []
        for i, ch in enumerate(chunks[:limit], start=1):
            txt = (ch.get("text") or "").replace("\n", " ")
            if len(txt) > show_chars:
                txt = txt[:show_chars] + "…"
            preview.append({
                "top": i,
                "source": ch.get("source"),
                "idx": ch.get("idx"),
                "snippet": txt
            })
        return {"total_chunks": len(chunks), "preview": preview}
    except Exception as e:
        return {"error": f"/kb/chunks 读取失败: {e}"}

# === 调试用：直接测 RAG 检索命中 ===
@app.get("/kb/search")
def kb_search(q: str, topk: int = 4, show_chars: int = 160):
    """
    直接调用检索器看看命中是否合理
    用法示例：/kb/search?q=积分兑换的商品是否可以开发票&topk=3
    """
    try:
        hits = retriever.retrieve(q, topk=topk)
        out = []
        for h in hits:
            txt = (h.get("text") or "").replace("\n", " ")
            if len(txt) > show_chars:
                txt = txt[:show_chars] + "…"
            out.append({
                "score": h.get("score"),
                "source": h.get("source"),
                "idx": h.get("idx"),
                "snippet": txt
            })
        return {"question": q, "hits_count": len(out), "hits": out}
    except Exception as e:
        return {"error": f"/kb/search 失败: {e}"}

@app.get("/kb/split_preview")
def kb_split_preview(
    file: str = Query(..., description="kb 目录下的文件名（例如 rules_daily_utf8.txt）"),
    level: str = Query("block", description="预览级别：para=只看段落；block=看打包块（不含重叠）"),
    show_chars: int = Query(160, description="每条预览展示多少字符"),
    limit: int = Query(30, description="最多显示多少条")
):
    """
    预览“切分结果”：
      - level=para  → 只看段落级别 (split_into_paragraphs)
      - level=block → 段落打包后的块（不含重叠）
    """
    path = Path(KB_DIR) / file
    if not path.exists():
        return {"ok": False, "error": f"文件不存在：{path}"}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        paras = debug_split_paragraphs_from_text(text)
        if level == "para":
            items = paras
        else:
            blocks = debug_pack_paragraphs_to_blocks(paras)
            items = blocks

        preview = []
        for i, item in enumerate(items[:limit], start=1):
            one = item.replace("\n", " ")
            if show_chars and len(one) > show_chars:
                one = one[:show_chars] + "…"
            preview.append({"top": i, "snippet": one})

        return {
            "ok": True,
            "file": file,
            "level": level,
            "total_items": len(items),
            "preview": preview
        }
    except Exception as e:
        return {"ok": False, "error": f"/kb/split_preview 失败: {e}"}


@app.get("/kb/chunk_fulltext")
def kb_chunk_fulltext(
    source: str = Query(..., description="kb 文件名（如 rules_daily_utf8.txt）"),
    idx: int = Query(..., description="块的 1-based 段号（与你命中里的 idx 一致）")
):
    """
    根据（source, idx）返回最终块的完整正文（就是 retriever 用的块文本）。
    用于：Coze 看到某个命中后，来这里查整个块的原文（不用再手翻 kb 文件）。
    """
    try:
        chunks = getattr(retriever, "chunks", [])
        for c in chunks:
            if c.get("source") == source and int(c.get("idx", -1)) == int(idx):
                return {
                    "ok": True,
                    "source": source,
                    "idx": idx,
                    "text": c.get("text", "")
                }
        return {"ok": False, "error": f"未找到：{source}#段{idx}"}
    except Exception as e:
        return {"ok": False, "error": f"/kb/chunk_fulltext 失败: {e}"}

@app.post("/ask_debug")
def ask_debug(req: AskReq):
    hits = retriever.retrieve(req.question, topk=req.topk)
    # 原样返回命中，便于你调bm25
    return JSONResponse({
        "hits": [
            {
                "rank": i+1,
                "score": h["score"],
                "source": h["source"],
                "idx": h["idx"],
                "text": (h["text"][:300] + "…") if len(h["text"]) > 300 else h["text"]
            } for i, h in enumerate(hits)
        ]
    }, media_type="application/json; charset=utf-8")