"""Chấm ĐẦY ĐỦ cả hai vế của rubric: truy xuất + câu trả lời của tác tử.

Khác với run_comparison.py (chỉ đo retrieval), script này nối LLM thật vào
KnowledgeBaseAgent rồi chấm câu trả lời bằng độ tương đồng với gold answer —
cùng thang 0.80 mà Vinh và Bá Quân dùng, để số liệu so sánh được.

    EMBEDDING_PROVIDER=gemini LLM_PROVIDER=gemini py benchmark/run_rag_eval.py
    EMBEDDING_PROVIDER=local  LLM_PROVIDER=deepseek py benchmark/run_rag_eval.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from src.agent import KnowledgeBaseAgent
from src.chunking import compute_similarity
from src.llm import get_chat
from benchmark.queries import QUERIES, STRATEGIES
from benchmark.run_comparison import build_store, load_corpus, CachedEmbedder

PASS_THRESHOLD = 0.80
NL = chr(10)


def get_embedder():
    p = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if p == "gemini":
        from src.embeddings import GeminiEmbedder
        return GeminiEmbedder()
    if p == "lexical":
        from benchmark.lexical_embedder import LexicalEmbedder
        return LexicalEmbedder().fit([b for _, b in load_corpus()])
    from src.embeddings import LocalEmbedder
    return LocalEmbedder()


def main() -> int:
    strat = os.getenv("STRATEGY", "C").upper()
    cfg = STRATEGIES[strat]
    emb = CachedEmbedder(get_embedder())
    llm = get_chat()

    print("=" * 84)
    print("CHẤM RAG ĐẦY ĐỦ — truy xuất + câu trả lời của tác tử")
    print("=" * 84)
    print(f"Chiến lược : {strat}. {cfg['chunker']} {cfg['params']}")
    print(f"Bộ nhúng   : {emb._backend_name}")
    print(f"LLM        : {llm._backend_name}")
    print(f"Ngưỡng đạt : tương đồng câu trả lời >= {PASS_THRESHOLD}")

    store, lengths = build_store(strat, emb)
    print(f"Corpus     : {len(load_corpus())} tài liệu -> {len(lengths)} chunk" + NL)

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    total, top1, top3, passed = 0, 0, 0, 0

    for q in QUERIES:
        res = store.search(q["question"], top_k=3)
        ranked = [r["metadata"]["doc_id"] for r in res]
        rank = ranked.index(q["gold_doc"]) + 1 if q["gold_doc"] in ranked else 0
        answer = agent.answer(q["question"], top_k=3)
        sim = compute_similarity(emb(answer), emb(q["gold_answer"]))
        ok = sim >= PASS_THRESHOLD
        score = 2 if (rank and ok) else (1 if rank else 0)
        total += score
        top1 += int(rank == 1)
        top3 += int(rank > 0)
        passed += int(ok)

        print(f"[{q['id']}] ({q['kind']}) {q['question']}")
        print(f"   Tài liệu đúng : {q['gold_doc']}  ->  hạng {rank or '-'}")
        print(f"   Gold          : {q['gold_answer']}")
        print(f"   Agent trả lời : {answer[:220]}")
        print(f"   Tương đồng    : {sim:.4f}  ->  {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        print(f"   Điểm câu này  : {score}/2" + NL)

    n = len(QUERIES)
    print("=" * 84)
    print("BẢNG ĐIỂM")
    print(f"   Top-1 hit               : {top1}/{n} ({top1/n:.0%})")
    print(f"   Top-3 hit               : {top3}/{n} ({top3/n:.0%})")
    print(f"   Câu trả lời đạt ngưỡng  : {passed}/{n} ({passed/n:.0%})")
    print(f"   TỔNG ĐIỂM TRUY XUẤT     : {total}/10")
    print(f"   Số lệnh gọi API nhúng   : {emb.calls}")
    print("=" * 84)

    qf = next((q for q in QUERIES if q.get("needs_filter")), None)
    if qf:
        print(NL + f"A/B METADATA FILTER — {qf['id']}: {qf['question']}")
        no_f = store.search(qf["question"], top_k=3)
        wi_f = store.search_with_filter(qf["question"], top_k=3, metadata_filter=qf["needs_filter"])
        for label, rs in (("KHÔNG lọc", no_f), ("CÓ lọc   ", wi_f)):
            print(f"  {label}:")
            for i, r in enumerate(rs, 1):
                m = r["metadata"]
                mark = " <== SAI ĐỐI TƯỢNG" if m.get("audience") != "student" else ""
                print(f"    {i}. score={r['score']:+.4f} | {m['doc_id']} | aud={m.get('audience')}{mark}")
        a_no = agent.answer(qf["question"], top_k=3)
        ctx = NL.join(r["content"] for r in wi_f)
        a_wi = llm(f"Context:{NL}{ctx}{NL}{NL}Question: {qf['question']}{NL}Answer:")
        print(f"{NL}  Agent KHÔNG lọc : {a_no[:200]}")
        print(f"  Agent CÓ lọc    : {a_wi[:200]}")
        print(f"{NL}  Gold: {qf['gold_answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
