"""Bộ nhúng TF-IDF thuần Python (từ vựng, không phải ngữ nghĩa).

Dùng khi chưa có sentence-transformers: cho tín hiệu truy xuất THẬT thay vì
nhiễu của MockEmbedder (vốn băm MD5 nên hai câu gần nghĩa vẫn ra vector khác hẳn).
IDF fit trên toàn văn 10 tài liệu gốc nên mọi chiến lược chunking dùng chung
một không gian vector -> so sánh công bằng.
"""
from __future__ import annotations
import math, re
from collections import Counter

TOKEN = re.compile(r"[0-9a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+")


def tokenize(text: str) -> list[str]:
    words = TOKEN.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]   # unigram + bigram


class LexicalEmbedder:
    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}
        self._backend_name = "lexical TF-IDF (unigram+bigram)"

    def fit(self, documents: list[str]) -> "LexicalEmbedder":
        n = len(documents)
        df = Counter()
        for d in documents:
            df.update(set(tokenize(d)))
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._default_idf = math.log(n + 1) + 1.0
        return self

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tf = Counter(tokenize(text))
        if not tf:
            return vec
        maxtf = max(tf.values())
        for tok, c in tf.items():
            w = (0.5 + 0.5 * c / maxtf) * self.idf.get(tok, self._default_idf)
            vec[hash(tok) % self.dim] += w
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
