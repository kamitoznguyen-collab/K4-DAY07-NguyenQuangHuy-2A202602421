"""Đối chứng LLM: GIỮ NGUYÊN retrieval, chỉ đổi mô hình sinh câu trả lời.

Cả nhóm có 3 bạn chạy LLM thật nhưng mỗi người một bộ truy xuất khác nhau, nên
không tách được "câu trả lời kém do retrieval hay do LLM". Script này cố định
ngữ cảnh top-3 rồi đưa CÙNG một ngữ cảnh cho nhiều LLM -> cô lập ảnh hưởng của LLM.

    py benchmark/run_llm_ab.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from src.chunking import compute_similarity
from src.embeddings import LocalEmbedder
from src.llm import GeminiChat, DeepSeekChat
from benchmark.queries import QUERIES, STRATEGIES
from benchmark.run_comparison import build_store, load_corpus, CachedEmbedder

PASS = 0.80
NL = chr(10)
STRAT = "C"


def main() -> int:
    emb = CachedEmbedder(LocalEmbedder())
    store, lengths = build_store(STRAT, emb)

    llms = {}
    for name, factory in (("Gemini", GeminiChat), ("DeepSeek", DeepSeekChat)):
        try:
            llms[name] = factory()
        except Exception as ex:
            print(f"[bo qua {name}] {type(ex).__name__}: {str(ex)[:100]}")

    print("=" * 88)
    print("ĐỐI CHỨNG LLM — cùng retrieval, khác mô hình sinh")
    print("=" * 88)
    print(f"Retrieval cố định : {STRATEGIES[STRAT]['chunker']} {STRATEGIES[STRAT]['params']}"
          f" + {emb._backend_name}  ({len(lengths)} chunk)")
    for n, m in llms.items():
        print(f"LLM {n:<10}    : {m._backend_name}")
    print(f"Ngưỡng đạt        : tương đồng >= {PASS}" + NL)

    totals = {n: 0 for n in llms}
    for q in QUERIES:
        res = store.search(q["question"], top_k=3)
        ranked = [r["metadata"]["doc_id"] for r in res]
        rank = ranked.index(q["gold_doc"]) + 1 if q["gold_doc"] in ranked else 0
        ctx = NL.join(r["content"] for r in res)
        prompt = f"Context:{NL}{ctx}{NL}{NL}Question: {q['question']}{NL}Answer:"

        print(f"[{q['id']}] {q['question'][:74]}")
        print(f"   tài liệu đúng hạng {rank or '-'}  |  gold: {q['gold_answer'][:70]}")
        for n, m in llms.items():
            try:
                ans = m(prompt)
            except Exception as ex:
                print(f"   {n:<9}: LOI {type(ex).__name__}"); continue
            sim = compute_similarity(emb(ans), emb(q["gold_answer"]))
            ok = sim >= PASS
            score = 2 if (rank and ok) else (1 if rank else 0)
            totals[n] += score
            print(f"   {n:<9}: sim={sim:.4f} {'ĐẠT ' if ok else 'TRƯỢT'} [{score}/2]  {ans[:96]}")
        print()

    print("=" * 88)
    print("BẢNG ĐIỂM — retrieval giống hệt nhau, khác biệt duy nhất là LLM")
    for n, s in totals.items():
        print(f"   {n:<10}: {s}/10")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
