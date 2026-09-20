"""Tính điểm cosine thực tế cho 5 cặp câu ở Mục 4 của REPORT_CANHAN.md.

Chạy SAU khi đã tự điền cột "Dự đoán" — bản chất bài tập là đoán trước rồi đối chiếu.

    EMBEDDING_PROVIDER=local   py benchmark/similarity_check.py   # MiniLM da ngu
    EMBEDDING_PROVIDER=gemini  py benchmark/similarity_check.py   # can GEMINI_API_KEY trong .env
    EMBEDDING_PROVIDER=mock    py benchmark/similarity_check.py   # bam MD5 - vo nghia, chi de doi chung
"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunking import compute_similarity

PAIRS = [
    ("Sinh viên được miễn 100% học phí",
     "Người học không phải nộp bất kỳ khoản học phí nào"),
    ("Học bổng Sigma Gold cấp 15 triệu đồng mỗi tháng",
     "Trợ cấp sinh hoạt phí hàng tháng cho sinh viên"),
    ("Điều kiện duy trì học bổng Đinh Thiện Lý",
     "Công thức nấu phở bò truyền thống Hà Nội"),
    ("Hồ sơ đăng ký học bổng Vallet gồm những gì",
     "Sinh viên cần nộp giấy tờ nào để xin học bổng Vallet"),
    ("Học bổng ngành Vi mạch bán dẫn tại HaUI",
     "Chính sách nội trú cho sinh viên dân tộc thiểu số"),
]


def get_embedder():
    from dotenv import load_dotenv
    load_dotenv(override=False)
    p = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if p == "local":
        from src.embeddings import LocalEmbedder
        return LocalEmbedder()
    if p == "gemini":
        from src.embeddings import GeminiEmbedder
        return GeminiEmbedder()
    if p == "openai":
        from src.embeddings import OpenAIEmbedder
        return OpenAIEmbedder()
    from src.embeddings import MockEmbedder
    return MockEmbedder()


def main() -> int:
    emb = get_embedder()
    print(f"Bo nhung: {getattr(emb, '_backend_name', '?')}\n")
    print(f"{'Cap':<5}{'Diem':>9}   Cau A  /  Cau B")
    print("-" * 78)
    for i, (a, b) in enumerate(PAIRS, 1):
        s = compute_similarity(emb(a), emb(b))
        print(f"{i:<5}{s:>9.4f}   {a}")
        print(f"{'':<14}{b}")
    print("\nDan cot 'Diem' vao bang Muc 4 trong report/REPORT_CANHAN.md,")
    print("roi doi chieu voi cot 'Du doan' ban da dien truoc do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
