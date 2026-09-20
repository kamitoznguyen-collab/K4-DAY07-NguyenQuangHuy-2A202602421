"""Chạy 3 chiến lược chunking trên cùng corpus + cùng 5 benchmark query, rồi so sánh."""
from __future__ import annotations
import io, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.store import EmbeddingStore
from src.models import Document
from benchmark.queries import QUERIES, STRATEGIES

CORPUS = Path(__file__).resolve().parent.parent / "data" / "scholarship"
CHUNKERS = {"fixed": FixedSizeChunker, "sentence": SentenceChunker, "recursive": RecursiveChunker}
NL = chr(10)


def load_corpus():
    docs = []
    for p in sorted(CORPUS.glob("*.md")):
        _, fm_raw, body = p.read_text(encoding="utf-8").split("---", 2)
        meta = {}
        for line in fm_raw.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        docs.append((meta, body.strip()))
    return docs


class CachedEmbedder:
    """Bọc embedder thật: nhớ kết quả theo nội dung + tự giãn nhịp theo hạn mức API.

    Free tier Gemini cho 100 lệnh embed/phút, nên cần throttle chủ động thay vì
    để server trả 429 rồi mới xử lý.
    """

    def __init__(self, inner, rpm: int = 0):
        import os
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self._backend_name = getattr(inner, "_backend_name", inner.__class__.__name__)
        self.calls = 0
        if not rpm:
            rpm = 90 if "gemini" in self._backend_name.lower() else 0
        self._rpm = rpm
        self._window: list[float] = []

    def _throttle(self) -> None:
        if not self._rpm:
            return
        import time
        now = time.monotonic()
        self._window = [t for t in self._window if now - t < 60.0]
        if len(self._window) >= self._rpm:
            wait = 60.0 - (now - self._window[0]) + 0.5
            if wait > 0:
                print(f"   [throttle] cham han muc, cho {wait:.0f}s...", flush=True)
                time.sleep(wait)
                now = time.monotonic()
                self._window = [t for t in self._window if now - t < 60.0]
        self._window.append(time.monotonic())

    def __call__(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        import time
        for attempt in range(6):
            try:
                self._throttle()
                self._cache[text] = self._inner(text)
                self.calls += 1
                return self._cache[text]
            except Exception as ex:
                if "429" not in str(ex) and "RESOURCE_EXHAUSTED" not in str(ex):
                    raise
                import re as _re
                m = _re.search(r"retry in ([\d.]+)s", str(ex))
                wait = float(m.group(1)) + 1 if m else 15 * (attempt + 1)
                print(f"   [429] cho {wait:.0f}s roi thu lai (lan {attempt + 1})...", flush=True)
                time.sleep(wait)
        raise RuntimeError("Van bi 429 sau 6 lan thu lai")


def build_store(strategy_key, embedder):
    cfg = STRATEGIES[strategy_key]
    chunker = CHUNKERS[cfg["chunker"]](**cfg["params"])
    store = EmbeddingStore(collection_name=f"strat_{strategy_key}", embedding_fn=embedder)
    lengths = []
    for meta, body in load_corpus():
        for i, chunk in enumerate(chunker.chunk(body)):
            if not chunk.strip():
                continue
            lengths.append(len(chunk))
            store.add_documents([Document(id=f"{meta['doc_id']}::{i}", content=chunk,
                                          metadata={**meta, "chunk_index": i})])
    return store, lengths


def evaluate(store):
    rows, hits, rr_sum = [], 0, 0.0
    for q in QUERIES:
        results = store.search(q["question"], top_k=3)
        ranked = [r["metadata"]["doc_id"] for r in results]
        rank = ranked.index(q["gold_doc"]) + 1 if q["gold_doc"] in ranked else 0
        hits += int(rank > 0)
        rr_sum += (1.0 / rank) if rank else 0.0
        ctx = NL.join(r["content"] for r in results)
        missing = [k for k in q["gold_keys"] if k not in ctx]
        rows.append({"id": q["id"], "rank": rank,
                     "cover": 1 - len(missing) / len(q["gold_keys"]),
                     "missing": missing,
                     "top1_score": results[0]["score"] if results else 0.0})
    n = len(QUERIES)
    return hits / n, rr_sum / n, rows


def main():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env', override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider == "local":
        from src.embeddings import LocalEmbedder
        embedder = LocalEmbedder()
    elif provider == "gemini":
        from src.embeddings import GeminiEmbedder
        embedder = GeminiEmbedder()
    elif provider == "lexical":
        from benchmark.lexical_embedder import LexicalEmbedder
        embedder = LexicalEmbedder().fit([b for _, b in load_corpus()])
    else:
        from src.embeddings import MockEmbedder
        embedder = MockEmbedder()

    embedder = CachedEmbedder(embedder)
    print(f"Bộ nhúng: {embedder._backend_name}")
    print(f"Corpus:   {len(load_corpus())} tài liệu" + NL)

    summary, stores = {}, {}
    for key in STRATEGIES:
        store, lengths = build_store(key, embedder)
        stores[key] = store
        hit, mrr, rows = evaluate(store)
        summary[key] = {"chunks": len(lengths),
                        "avg_len": sum(lengths) / len(lengths) if lengths else 0,
                        "hit": hit, "mrr": mrr, "rows": rows,
                        "cover": sum(r["cover"] for r in rows) / len(rows)}

    print("=" * 86)
    print(f"{'Chiến lược':<34}{'#chunk':>8}{'dài TB':>9}{'Hit@3':>8}{'MRR':>7}{'Phủ đáp án':>13}")
    print("=" * 86)
    for key, cfg in STRATEGIES.items():
        s = summary[key]
        print(f"{key}. {cfg['owner']} — {cfg['chunker']:<14}"
              f"{s['chunks']:>8}{s['avg_len']:>9.0f}{s['hit']:>7.0%}{s['mrr']:>7.2f}{s['cover']:>12.0%}")

    print(NL + "Thứ hạng tài liệu đúng trong top-3 (- = trượt):")
    print(f"{'':<34}" + "".join(f"{q['id']:>6}" for q in QUERIES))
    for key, cfg in STRATEGIES.items():
        print(f"{key}. {cfg['owner']:<31}" +
              "".join(f"{(r['rank'] or '-'):>6}" for r in summary[key]["rows"]))

    print(NL + "Mẩu số liệu BỊ MẤT khỏi ngữ cảnh top-3 (chunk cắt đứt đáp án):")
    for key, cfg in STRATEGIES.items():
        miss = {r["id"]: r["missing"] for r in summary[key]["rows"] if r["missing"]}
        print(f"  {key}. {cfg['owner']:<14} " + (str(miss) if miss else "(không mất gì)"))

    q1 = next(q for q in QUERIES if q.get("needs_filter"))
    print(NL + "=" * 86)
    print(f"A/B metadata filter — {q1['id']}: {q1['question']}")
    print("=" * 86)
    for key, cfg in STRATEGIES.items():
        store = stores[key]
        no_f = [r["metadata"]["doc_id"] for r in store.search(q1["question"], top_k=3)]
        with_f = [r["metadata"]["doc_id"] for r in
                  store.search_with_filter(q1["question"], top_k=3, metadata_filter=q1["needs_filter"])]
        tag = "ĐỔI KẾT QUẢ" if no_f != with_f else "không đổi"
        leak = sum(1 for d in no_f if d == q1["distractor_doc"])
        print(f"{NL}{key}. {cfg['owner']}  [{tag}]  — tài liệu sai đối tượng lọt top-3: {leak}")
        print(f"   không lọc  : {no_f}")
        print(f"   lọc student: {with_f}")
    print(f"{NL}Gold: {q1['gold_answer']}")
    print(f"Bẫy:  {q1['why_filter']}")


if __name__ == "__main__":
    main()
