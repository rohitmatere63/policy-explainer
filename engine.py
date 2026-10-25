"""Everything except the UI: PDF -> clauses -> BM25 search -> grounded answer -> verification."""
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
from rank_bm25 import BM25Plus

import sarvam as sv

DISCLAIMER = ("AI generated explanation for information only. It is not insurance advice and does not "
              "modify the policy. The policy wording and the insurer's decision prevail.")
FLAG_MSG = {
    "not_in_document": "This policy document does not state this. Please check with the insurer.",
    "conflict": "Two parts of the policy appear to disagree. Please confirm with the insurer.",
    "ambiguous": "This clause is vague or depends on another document or the insurer's discretion.",
    "needs_external": "This depends on information outside the policy (network, premium, or your medical history).",
    "low_confidence": "I am not fully sure about this answer. Please verify with the insurer.",
}
SUMMARY_QUESTIONS = {
    "Sum insured": "What is the sum insured and what are the cover limits?",
    "Waiting periods": "What are the waiting periods?",
    "Pre-existing diseases": "How are pre-existing diseases covered and after how long?",
    "Room rent & sub-limits": "Is there a room rent limit or any sub-limit?",
    "Co-payment & deductible": "Is there any co-payment or deductible?",
    "Major exclusions": "What are the main exclusions?",
    "Claim process": "How do cashless and reimbursement claims work and what are the time limits?",
    "Renewal & portability": "What are the renewal, grace period and portability terms?",
}
STOP = set("a an the is are was be of to in on for and or if do does can i my me what how when which this that "
           "it with by as at from any there their under policy cover covered covers document about tell explain".split())
OVERVIEW_RE = re.compile(r"\b(about|overview|summar\w*|what is this|explain this|describe)\b", re.I)

CLAUSE_RE = re.compile(r"^\s*((?:\d+\.)+\d*|\d+[.)]|\([a-z0-9ivx]{1,4}\)|[A-Z]\.)\s+\S")
HEADING_RE = re.compile(r"^\s*(?:(?:section|part|annexure|schedule)\b.*|[A-Z][A-Z &/,\-()]{6,})\s*$", re.I)


def tok(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


# ---------- ingestion ----------
class Policy:
    def __init__(self, pdf: bytes):
        self.pdf = pdf
        doc = fitz.open(stream=pdf, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        self.n_pages = len(pages)
        self.chunks = self._chunk(pages)
        self.bm25 = BM25Plus([tok(f"{c['section']} {c['text']}") for c in self.chunks]) if self.chunks else None
        self.low_quality = (sum(len(p) for p in pages) / max(len(pages), 1) < 300) or len(self.chunks) < 5

    @staticmethod
    def _chunk(pages: list[str]) -> list[dict]:
        counts = Counter(l.strip() for p in pages for l in set(p.splitlines()) if l.strip())
        junk = {l for l, n in counts.items() if n >= max(3, int(0.4 * len(pages))) and len(l) < 120}
        chunks, section = [], ""

        def flush(buf, page, clause):
            text = " ".join(buf).strip()
            for i in range(0, len(text), 1300) if len(text) >= 25 else []:
                chunks.append({"id": f"p{page}-c{len(chunks) + 1}", "page": page, "section": section,
                               "clause": clause, "text": text[i:i + 1300]})

        for pno, raw in enumerate(pages, 1):
            buf, clause = [], ""
            for line in raw.splitlines():
                if not line.strip() or line.strip() in junk:
                    continue
                if HEADING_RE.match(line) and len(line) < 90:
                    flush(buf, pno, clause); buf, clause = [], ""
                    section = line.strip().title()
                elif (m := CLAUSE_RE.match(line)):
                    flush(buf, pno, clause); buf, clause = [line.strip()], m.group(1)
                else:
                    buf.append(line.strip())
            flush(buf, pno, clause)
        return chunks

    # ---------- retrieval ----------
    def search(self, q: str, top: int = 5) -> tuple[list[dict], float]:
        """Returns (top chunks, coverage 0..1 = share of question keywords found in them)."""
        qt = tok(q)
        if not qt or not self.bm25:
            return [], 0.0
        scores = self.bm25.get_scores(qt)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top]
        hits = [self.chunks[i] for i in idx]
        have = set().union(*(set(tok(c["text"] + " " + c["section"])) for c in hits[:3]))
        return hits, sum(t in have for t in set(qt)) / len(set(qt))

    def overview_chunks(self, n: int = 8) -> list[dict]:
        """Broad questions ('what is this document about'): first clause of each section, spread over the document."""
        firsts, seen = [], set()
        for c in self.chunks:
            if c["section"] not in seen:
                seen.add(c["section"]); firsts.append(c)
        if len(firsts) > n:
            step = len(firsts) / n
            firsts = [firsts[int(i * step)] for i in range(n)]
        return (self.chunks[:2] + [c for c in firsts if c not in self.chunks[:2]])[:n + 2]

    def definitions_and_exclusions(self, q: str, n: int = 2) -> list[dict]:
        pool = [c for c in self.chunks if re.search(r"definition|exclusion", c["section"], re.I)]
        qt = set(tok(q))
        return sorted(pool, key=lambda c: -len(qt & set(tok(c["text"]))))[:n]

    def page_png(self, page: int, quote: str = "") -> bytes:
        doc = fitz.open(stream=self.pdf, filetype="pdf")
        pg, words = doc[page - 1], quote.split()
        for size in (12, 8, 5):
            rects = pg.search_for(" ".join(words[:size])) if words else []
            for r in rects:
                pg.add_highlight_annot(r)
            if rects:
                break
        return pg.get_pixmap(dpi=100).tobytes("png")


# ---------- verification ----------
def _norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def _nums(s): return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)}


def verify(item: dict, cmap: dict) -> dict | None:
    """Keep a statement only if its citation exists, its quote is really in the clause, and numbers match."""
    cites = [c for c in item.get("cites", []) if c in cmap]
    text, quote = str(item.get("text", "")).strip(), str(item.get("quote", "")).strip()
    if not (text and cites and quote):
        return None
    src = " ".join(cmap[c]["text"] for c in cites)
    qn, sn = _norm(quote), _norm(src)
    if qn not in sn:  # tolerate small wording/line-break differences: 85% of quote words must be in the clause
        qw, sw = qn.split(), set(sn.split())
        if len(qw) < 4 or sum(w in sw for w in qw) / len(qw) < 0.85:
            return None
    if not _nums(text) <= _nums(src):
        return None
    return {"text": text, "cites": cites, "quote": quote}


# ---------- grounded answer ----------
SYSTEM = """You are a health insurance policy explainer. Use ONLY the clauses provided. Never use outside knowledge.
Never recommend buying/rejecting a policy and never predict claim approval.
Reply with ONE JSON object and nothing else:
{"answerable": true|false,
 "short_answer": "1-2 plain-English sentences",
 "statements": [{"text": "explanation sentence", "cites": ["<clause id>"], "quote": "verbatim words (max 25) copied from that clause"}],
 "conditions": [{"text": "an 'it depends on' condition, limit or exclusion", "cites": ["<clause id>"], "quote": "verbatim words from the clause"}],
 "conflict": true|false, "ambiguous": true|false, "needs_external": true|false}
Copy numbers exactly. If the clauses do not answer the question, set answerable=false.
conflict=true if two clauses disagree. ambiguous=true for vague/discretionary wording or references to other documents.
needs_external=true if it depends on the insurer's network, premium or the person's medical history."""


def answer(policy: Policy, q_en: str, lang: str, min_cov: float = 0.4) -> dict:
    if OVERVIEW_RE.search(q_en) and len(tok(q_en)) <= 3:
        hits, cov = policy.overview_chunks(), 1.0
        ctx = hits
    else:
        hits, cov = policy.search(q_en)
        ctx = hits + [c for c in policy.definitions_and_exclusions(q_en) if c not in hits]
    dbg = {"coverage": round(cov, 2), "retrieved": [f"{c['id']} (p{c['page']})" for c in ctx]}
    cmap = {c["id"]: c for c in ctx}
    stmts, conds, short, flags, ratio = [], [], "", [], 0.0

    if cov >= min_cov:
        body = "\n\n".join(f"[id={c['id']} | page {c['page']} | {c['section']} | clause {c['clause'] or '-'}]\n{c['text']}" for c in ctx)
        a = sv.chat_json(SYSTEM, f"CLAUSES:\n{body}\n\nQUESTION: {q_en}")
        dbg["llm_json_ok"] = a is not None
        dbg["llm_answerable"] = bool(a and a.get("answerable"))
        if a:
            dbg["raw_statements"] = len(a.get("statements", []))
        if a and a.get("answerable"):
            raw = a.get("statements", [])
            stmts = [v for s in raw if isinstance(s, dict) and (v := verify(s, cmap))]
            conds = [v for s in a.get("conditions", []) if isinstance(s, dict) and (v := verify(s, cmap))]
            ratio = len(stmts) / max(len(raw), 1)
            dbg["statements_verified"] = f"{len(stmts)}/{len(raw)}"
            short = str(a.get("short_answer", ""))
            if not _nums(short) <= _nums(" ".join(c["text"] for c in ctx)):
                short = ""  # numbers in the headline must exist in the retrieved text
            flags += [f for f in ("conflict", "ambiguous", "needs_external") if a.get(f)]

    conf = round(0.5 * cov + 0.5 * ratio, 2) if stmts else 0.0
    if not stmts:
        flags = ["not_in_document"]
    elif conf < 0.5:
        flags.append("low_confidence")

    seen, cites = set(), []
    for s in stmts + conds:
        for c in s["cites"]:
            if c not in seen:
                seen.add(c)
                cites.append({"page": cmap[c]["page"], "section": cmap[c]["section"],
                              "clause": cmap[c]["clause"], "quote": s["quote"]})
    # translate every user-facing string in parallel (was one-by-one = slow)
    texts = [short] + [s["text"] for s in stmts] + [s["text"] for s in conds] + [FLAG_MSG[f] for f in flags]
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda t: sv.translate(t, lang) if t else t, texts))
    n1, n2 = 1 + len(stmts), 1 + len(stmts) + len(conds)
    return {"short": out[0],
            "statements": [{"text": out[1 + i], "cites": s["cites"]} for i, s in enumerate(stmts)],
            "conditions": out[n1:n2], "flags": out[n2:], "flag_ids": flags,
            "citations": cites, "confidence": conf, "debug": dbg}
