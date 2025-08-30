# rag_step1_bm25.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 依赖 ---
import os, re, glob, datetime
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# ===================== 配置 =====================
USE_SEMANTIC = True   # 设为 False 时仅用 BM25
if USE_SEMANTIC:
    _sem = SentenceTransformer("BAAI/bge-small-zh-v1.5")
else:
    _sem = None

KB_DIR = r"D:\Program Files\PythonProjects\PythonProject\kb"

# 每块目标字数/重叠字数（保持你的习惯）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# 自定义词典：避免破坏业务词
custom_words = [
    "在籍续约","提前续约","效期生效","积分",
    "PLUS会员","生活服务包","年卡","折算积分"
]
for w in custom_words:
    jieba.add_word(w, freq=100)

# 停用词 / 同义词归一
STOPWORDS = ["您好","辛苦","核实","客户","反馈","用户名","问题：","请问","谢谢","麻烦","一下","表示"]
SYNONYMS = {
    "续约": "续费",
    "开通": "开卡",
    "发放": "下发",
    "发": "下发",
    "优惠卷": "优惠券"
}

# “必须包含/奖励/惩罚关键词”占位（保留你原有接口）
MUST_ANY_LEFT  = []
MUST_ANY_RIGHT = []
CORE_KEYWORDS = []
PAIR_BONUS    = []
PENALTY_KEYWORDS = []


# ===================== 工具函数 =====================
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-9) / (np.linalg.norm(b) + 1e-9))

def rerank_semantic(query, candidates, topk=4):
    if not USE_SEMANTIC or _sem is None:
        return [i for i, _, _ in candidates[:topk]]
    q = normalize_text(query)
    q_emb = _sem.encode([q], normalize_embeddings=True)[0]
    d_emb = _sem.encode([normalize_text(t) for _, t, _ in candidates], normalize_embeddings=True)
    rescored = []
    for (i, t, bm25_s), e in zip(candidates, d_emb):
        rescored.append((i, 0.4 * bm25_s + 0.6 * _cos(q_emb, e)))  # 混合权重
    rescored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in rescored[:topk]]

def clean_text(s: str) -> str:
    s = s.replace("\uFEFF", "").replace("\u200b", "")
    s = re.sub(r"\?{3,}", "", s)
    s = re.sub(r"[·•◦\t]+", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def normalize_text(s: str) -> str:
    for sw in STOPWORDS:
        s = s.replace(sw, " ")
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。；：、\-\(\)（）/ ]+"," ", s)
    for k,v in SYNONYMS.items():
        s = s.replace(k,v)
    s = re.sub(r"\s{2,}"," ", s).strip()
    return s

def normalize_query(q: str) -> str:
    for sw in STOPWORDS:
        q = q.replace(sw, " ")
    q = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。；：、\-\(\)（）/ ]+", " ", q)
    for k, v in SYNONYMS.items():
        q = q.replace(k, v)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


# ===================== 段落切分（关键改造） =====================
_Q_PAT = re.compile(r"^\s*(?:Q:|Q：|问:|问：)\s*")
_A_PAT = re.compile(r"^\s*(?:A:|A：|答:|答：)\s*")

# 编号/标题行：1. / 1.1 / 2.3.4 / 1、 / 1) / （一） / 一、 等
_HEADING_PATS = [
    re.compile(r"^\s*\d+(?:\.\d+)*[、\.)]?\s+"),         # 1. / 1.1 / 1.2.3 / 1、 / 1) / 1.)
    re.compile(r"^\s*[（(][一二三四五六七八九十]+[)）]\s*"), # （一）（二）
    re.compile(r"^\s*[一二三四五六七八九十]+、\s*"),         # 一、二、三、
    re.compile(r"^\s*[-•·*]\s+"),                        # - • · *
]

_SENT_SPLIT = re.compile(r"(?<=[。；！!？\?])")  # 句末分句

def _is_heading(line: str) -> bool:
    return any(p.match(line) for p in _HEADING_PATS)

def _is_blank(line: str) -> bool:
    return len(line.strip()) == 0

def split_into_paragraphs(text: str):
    """
    目标：
      1) 优先按“空行 / 标题编号行”切段
      2) 若遇到 Q: 开头，则用状态机把 Q + A + 后续解释 合成一个完整块（直到下一个 Q: 或新的标题/空行）
      3) 超长段再按句号/分号等断句拼块，尽量不切断句子
    """
    text = clean_text(text)
    lines = text.split("\n")

    parts = []
    cur = []               # 普通段缓冲
    qa_buf = []            # Q&A 段缓冲
    in_qa = False
    seen_a = False

    def _flush_cur():
        nonlocal cur
        if cur:
            s = "\n".join(cur).strip()
            if s:
                parts.append(s)
        cur = []

    def _flush_qa():
        nonlocal qa_buf, in_qa, seen_a
        if qa_buf:
            s = "\n".join(qa_buf).strip()
            if s:
                parts.append(s)
        qa_buf = []
        in_qa = False
        seen_a = False

    for ln in lines:
        # 命中新的标题/编号：切断当前缓冲
        if _is_heading(ln):
            if in_qa:
                _flush_qa()
            _flush_cur()
            cur.append(ln)
            continue

        # 空行：结束一个逻辑段
        if _is_blank(ln):
            if in_qa:
                _flush_qa()
            else:
                _flush_cur()
            continue

        # Q/A 状态机
        if _Q_PAT.match(ln):
            # 新的 Q 来了：切掉上一段（无论上一段是否普通/QA）
            if in_qa:
                _flush_qa()
            else:
                _flush_cur()
            qa_buf = [ln]
            in_qa = True
            seen_a = False
            continue

        if in_qa:
            if _Q_PAT.match(ln):
                # 极端：Q 后又来了一个 Q（上一问没 A），也切段
                _flush_qa()
                qa_buf = [ln]
                in_qa = True
                seen_a = False
                continue
            if _A_PAT.match(ln):
                seen_a = True
                qa_buf.append(ln)
                continue
            # 既不是新 Q 也不是 A：就作为 Q/A 的补充说明
            qa_buf.append(ln)
            continue

        # 普通行：拼进当前段
        cur.append(ln)

    # 文件结束，收尾
    if in_qa:
        _flush_qa()
    else:
        _flush_cur()

    # 二次处理：把过长的 parts 再“按句子拼接”成 <= CHUNK_SIZE
    paragraphs = []
    for p in parts:
        if len(p) <= CHUNK_SIZE:
            paragraphs.append(p)
        else:
            sent = _SENT_SPLIT.split(p)
            buf = ""
            for s in sent:
                if len(buf) + len(s) <= CHUNK_SIZE:
                    buf += s
                else:
                    if buf:
                        paragraphs.append(buf.strip())
                    buf = s
            if buf:
                paragraphs.append(buf.strip())

    return paragraphs


# ===================== KB 读取与打包 =====================
def read_kb_chunks():
    """
    读取 kb/*.txt：
      1) 先用 split_into_paragraphs 做“结构化段落”切分（不切断句子/不切断 Q&A）
      2) 在不破坏上一步边界的前提下，打包成 <= CHUNK_SIZE 的块
      3) 对相邻块做字符级“尾部重叠” CHUNK_OVERLAP（仅作为上下文桥，不改变边界含义）
    """
    chunks = []
    for path in glob.glob(os.path.join(KB_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = clean_text(text)
        paras = split_into_paragraphs(text)

        # 1) 段落打包（不打断结构化段）
        blocks = []
        cur, cur_len = "", 0
        for para in paras:
            if not cur:
                cur = para
                cur_len = len(para)
            elif cur_len + 2 + len(para) <= CHUNK_SIZE:
                cur = cur + "\n\n" + para
                cur_len = len(cur)
            else:
                blocks.append(cur.strip())
                cur = para
                cur_len = len(para)
        if cur:
            blocks.append(cur.strip())

        # 2) 滑动重叠：保留上一块的末尾 CHUNK_OVERLAP 字符，接到下一块开头
        for i, blk in enumerate(blocks):
            chunks.append({
                "text": blk,
                "source": os.path.basename(path),
                "idx": i + 1
            })
            if i < len(blocks) - 1 and CHUNK_OVERLAP > 0:
                overlap = blk[-CHUNK_OVERLAP:] if len(blk) > CHUNK_OVERLAP else blk
                blocks[i + 1] = overlap + "\n\n" + blocks[i + 1]

    return chunks

# ======== 调试辅助：暴露分段与打包 ========

def debug_split_paragraphs_from_text(text: str):
    """
    仅做“段落级”切分，不做打包/重叠，用于预览 split_into_paragraphs 的结果。
    """
    return split_into_paragraphs(clean_text(text))


def debug_pack_paragraphs_to_blocks(paragraphs: list[str], chunk_size: int = None):
    """
    把段落按 CHUNK_SIZE 打包成“块”（不做重叠）并返回，便于你在接口里预览“块层”的结果。
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    blocks = []
    cur, cur_len = "", 0
    for para in paragraphs:
        if not cur:
            cur = para
            cur_len = len(para)
        elif cur_len + 2 + len(para) <= chunk_size:
            cur = cur + "\n\n" + para
            cur_len = len(cur)
        else:
            blocks.append(cur.strip())
            cur = para
            cur_len = len(para)
    if cur:
        blocks.append(cur.strip())
    return blocks

# ===================== 检索器 =====================
class RetrieverBM25:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [list(jieba.cut(c["text"])) for c in chunks]
        k1 = float(os.getenv("BM25_K1", "1.5"))
        b  = float(os.getenv("BM25_B", "0.75"))
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def retrieve(self, query, topk=4):
        q_norm = normalize_text(query)
        q_tokens = list(jieba.cut(q_norm))

        base_scores = self.bm25.get_scores(q_tokens)

        # 必要词过滤（保持你的逻辑）
        pool_both, pool_either = [], []
        for i, c in enumerate(self.chunks):
            t = normalize_text(c["text"])
            has_left  = any(k in t for k in MUST_ANY_LEFT) if MUST_ANY_LEFT else True
            has_right = any(k in t for k in MUST_ANY_RIGHT) if MUST_ANY_RIGHT else True
            if has_left and has_right:
                pool_both.append(i)
            if has_left or has_right:
                pool_either.append(i)

        # 候选集放宽
        idx_pool = pool_both
        if len(idx_pool) < topk:
            idx_pool = pool_either
        if len(idx_pool) < topk:
            idx_pool = list(range(len(self.chunks)))

        topk = min(topk, len(idx_pool))

        # 业务加权
        scored = []
        for i in idx_pool:
            t_doc = normalize_text(self.chunks[i]["text"])
            bonus = 0.0
            for kw, w in CORE_KEYWORDS:
                if kw in t_doc: bonus += w
            for a, b, w in PAIR_BONUS:
                if (a in t_doc) and (b in t_doc): bonus += w
            for kw, w in PENALTY_KEYWORDS:
                if kw in t_doc: bonus -= w
            scored.append((i, base_scores[i] + bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        N = min(30, len(scored))
        candidates = [(i, self.chunks[i]["text"], base_scores[i]) for i, _ in scored[:N]]

        if USE_SEMANTIC and _sem is not None and len(candidates) > 0:
            final_idxs = rerank_semantic(q_norm, candidates, topk=topk)
        else:
            final_idxs = [i for i, _ in scored[:topk]]

        results = []
        for i in final_idxs:
            c = self.chunks[i]
            results.append({
                "score": round(float(base_scores[i]), 3),
                "text": c["text"],
                "source": c["source"],
                "idx": c["idx"],
            })
        return results


# ===================== 调试 & 导出 =====================
def pretty_print_hits(hits, show_chars=None):
    print("\n===== 命中明细（完整片段）=====")
    for i, h in enumerate(hits, 1):
        print(f"[Top{i}] 来源：{h['source']}#段{h['idx']} | 分数={h['score']}")
        text = h["text"].replace("\r", " ").replace("\n", " ")
        if show_chars is None:
            print(text)
        else:
            print(text[:show_chars] + ("…" if len(text) > show_chars else ""))
        print("-" * 60)

def export_hits_to_txt(query, hits, out_path=None):
    if out_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"hits_{ts}.txt"
    lines = []
    lines.append(f"【问题】{query}\n")
    for i, h in enumerate(hits, 1):
        lines.append(f"[Top{i}] 来源：{h['source']}#段{h['idx']} | 分数={h['score']}")
        lines.append(h["text"])
        lines.append("\n" + "=" * 80 + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已导出命中结果到：{out_path}")

def get_retriever():
    chunks = read_kb_chunks()
    print(f"[loader] 知识块加载完成：{len(chunks)} 段")
    return RetrieverBM25(chunks)


# ===================== 直接运行自测 =====================
if __name__ == "__main__":
    chunks = read_kb_chunks()
    print(f"已加载知识块：{len(chunks)} 段")

    retriever = RetrieverBM25(chunks)

    queries = [
        "您好用户名：jd_66d0a9851510b客户进线询问她的会员明天过期她兑换了洗车服务问洗车服务会随着会员过期而无法使用吗？",
        "积分兑换的商品是否可以开发票",
        "洗车多久过期",
        "Q：为什么积分只有9分？A：……"
    ]
    for q in queries:
        q_norm = normalize_text(q)
        print("\n【问题】", q)
        print("[debug] 归一后：", q_norm)
        hits = retriever.retrieve(q_norm, topk=4)
        for h in hits:
            print(f"- 命中：{h['source']}#段{h['idx']}  分数={h['score']}")
            print("  片段：", h["text"][:120].replace("\n", " "), "…")

    # 导出示例
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"hits_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            q_norm = normalize_text(q)
            f.write(f"【问题】{q}\n")
            hits = retriever.retrieve(q_norm, topk=4)
            for i, h in enumerate(hits, 1):
                text_full = h["text"].replace("\r", " ").replace("\n", " ")
                f.write(f"[Top{i}] 来源：{h['source']}#段{h['idx']} | 分数={h['score']}\n")
                f.write(text_full + "\n" + "=" * 80 + "\n")
            f.write("\n")
    print(f"已导出命中结果到：{out_path}")