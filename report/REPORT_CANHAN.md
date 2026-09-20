# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Quang Huy
**MSSV:** 2A202602421
**Nhóm:** K4-L3A — Nhóm Học Bổng Đại Học
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm nộp chung trong `REPORT_NHOM.md`. Thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Cosine đo **góc** giữa hai vector chứ không đo khoảng cách. Giá trị càng gần 1.0 thì hai vector càng cùng hướng, nghĩa là hai đoạn văn bản mang ý nghĩa tương đồng — bất kể chúng dài ngắn khác nhau hay dùng từ vựng khác nhau. Giá trị quanh 0 nghĩa là hai nội dung gần như không liên quan; giá trị âm nghĩa là chúng trái hướng nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Hồ sơ đăng ký học bổng Vallet gồm những gì"
- Câu B: "Sinh viên cần nộp giấy tờ nào để xin học bổng Vallet"
- Tại sao tương đồng: Hai câu hỏi **cùng một việc** — danh mục giấy tờ cần nộp — chỉ khác cách diễn đạt ("hồ sơ" và "giấy tờ", "đăng ký" và "xin"). Đo thực tế ở Mục 4 cho **0.8361**, cao nhất trong 5 cặp.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Điều kiện duy trì học bổng Đinh Thiện Lý"
- Câu B: "Công thức nấu phở bò truyền thống Hà Nội"
- Tại sao khác: Hai câu thuộc hai miền hoàn toàn tách biệt (quy định học vụ và ẩm thực), không chia sẻ chủ đề lẫn từ vựng. Đo thực tế cho **0.1229**.

**Tại sao cosine similarity được ưu tiên hơn khoảng cách Euclid cho text embeddings?**
> Vì khoảng cách Euclid bị chi phối bởi **độ lớn (magnitude)** của vector, mà độ lớn lại phụ thuộc độ dài văn bản. Một đoạn quy định dài 2.000 ký tự và một câu hỏi 20 ký tự cùng nói về học bổng Vallet sẽ có vector chênh lệch độ lớn rất nhiều, khiến Euclid báo là "xa nhau" dù cùng chủ đề. Cosine bỏ qua độ lớn, chỉ xét hướng, nên phản ánh đúng tương đồng ngữ nghĩa. Trong `EmbeddingStore.search` của tôi, các backend đều trả vector đã chuẩn hoá độ dài đơn vị, nên tôi dùng thẳng tích vô hướng — tương đương cosine mà bỏ được phép chia.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `ceil((độ_dài - overlap) / (chunk_size - overlap))`
> `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)` = **23 chunks**
>
> Kiểm chứng bằng code của tôi:
> ```python
> from src.chunking import FixedSizeChunker
> len(FixedSizeChunker(500, 50).chunk("x" * 10000))   # -> 23
> ```

**Nếu overlap tăng lên 100, số chunk thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75)` = **25 chunks**, tăng 2 chunk.
>
> Lý do muốn overlap lớn hơn: mỗi chunk "nhìn thấy" một phần nội dung của chunk kề bên, nên thông tin nằm ngay ranh giới cắt không bị mất. Thí nghiệm của tôi ở Mục 5 cho thấy đúng rủi ro này: câu Q1 hỏi mức học bổng ngành Vi mạch bán dẫn, chuỗi `4.200.000` bị **cả ba chiến lược** cắt rời khỏi tên ngành nên không chunk nào chứa trọn cặp (tên ngành + số tiền).

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex lookbehind `r'(?<=[.!?])\s+|(?<=\.)\n'` để tách tại ranh giới câu mà **vẫn giữ dấu câu** ở cuối câu trước (lookbehind không "nuốt" ký tự khớp). Sau khi tách, lọc bỏ chuỗi rỗng và strip khoảng trắng, rồi gom tuần tự theo `max_sentences_per_chunk`. Edge case đã xử lý: text rỗng trả về `[]`; text không có dấu chấm nào thì `re.split` trả về đúng một phần tử nên vẫn ra một chunk hợp lệ.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Đệ quy theo danh sách separator giảm dần độ "thô": `["\n\n", "\n", ". ", " ", ""]`. Base case là khi đoạn hiện tại đã `<= chunk_size` **hoặc** đã hết separator. Nếu separator hiện tại không xuất hiện trong text thì bỏ qua, thử separator kế tiếp — tránh đệ quy vô ích. Sau khi tách, thuật toán **gộp ngược** các mảnh liền kề vào một buffer cho tới sát `chunk_size`, để không sinh ra hàng loạt chunk vài ký tự khi tài liệu có nhiều dòng ngắn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `_make_record()` chuẩn hoá mỗi `Document` thành dict gồm `id`, `content`, `metadata` và `embedding` tính sẵn tại thời điểm thêm — nhúng một lần, tái dùng cho mọi truy vấn. `search` nhúng câu hỏi rồi tính **tích vô hướng** với từng vector đã lưu; vì các backend đều trả vector đã chuẩn hoá độ dài đơn vị nên dot product tương đương cosine, tiết kiệm được phép chia. Sắp xếp giảm dần và cắt `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Chọn **tiền lọc (pre-filter)** chứ không hậu lọc: lọc theo metadata trước rồi mới xếp hạng trên tập còn lại. Lý do: nếu xếp hạng trước rồi mới loại, `top_k` sẽ bị các chunk sai đối tượng chiếm chỗ và trả về ít hơn `k` kết quả hợp lệ. `delete_document` dựng lại list bằng comprehension, loại bỏ record có `id` **hoặc** `metadata['doc_id']` khớp, rồi so sánh độ dài trước/sau để trả `True`/`False` — không cần đếm thủ công.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba bước RAG: (1) `store.search(question, top_k)` lấy ngữ cảnh; (2) ghép các chunk thành khối `Context:` rồi đặt câu hỏi phía dưới; (3) đẩy prompt vào `llm_fn`. Việc tiêm ngữ cảnh **trước** câu hỏi là có chủ ý — mô hình đọc dữ liệu rồi mới gặp yêu cầu, giảm khả năng trả lời theo trí nhớ nội tại thay vì theo tài liệu.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```text
..........................................                               [100%]
42 passed in 0.07s
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Quy ước: **cao ≥ 0.5**, **thấp < 0.5**. Cột "Dự đoán" được ghi trước khi chạy đo.
Bộ nhúng: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
Lệnh đo: `EMBEDDING_PROVIDER=local py benchmark/similarity_check.py`

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được miễn 100% học phí | Người học không phải nộp bất kỳ khoản học phí nào | cao | 0.5734 | ✅ |
| 2 | Học bổng Sigma Gold cấp 15 triệu đồng mỗi tháng | Trợ cấp sinh hoạt phí hàng tháng cho sinh viên | cao | **0.2454** | ❌ |
| 3 | Điều kiện duy trì học bổng Đinh Thiện Lý | Công thức nấu phở bò truyền thống Hà Nội | thấp | 0.1229 | ✅ |
| 4 | Hồ sơ đăng ký học bổng Vallet gồm những gì | Sinh viên cần nộp giấy tờ nào để xin học bổng Vallet | cao | 0.8361 | ✅ |
| 5 | Học bổng ngành Vi mạch bán dẫn tại HaUI | Chính sách nội trú cho sinh viên dân tộc thiểu số | thấp | 0.2778 | ✅ |

**Dự đoán đúng: 4/5.**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là **cặp 2** — tôi dự đoán cao nhưng chỉ được **0.2454**. Hai câu mô tả cùng một loại sự vật: một khoản tiền trả hàng tháng cho sinh viên. Tôi nghĩ mô hình sẽ nhận ra "15 triệu đồng mỗi tháng" và "trợ cấp sinh hoạt phí hàng tháng" thuộc cùng phạm trù.
>
> Nhưng mô hình không suy ra được mối liên hệ ấy. So sánh với cặp 4 (0.8361) thì thấy rõ ranh giới: cặp 4 có chung thực thể hiển ngôn "học bổng Vallet" và chung khung câu hỏi "cần nộp gì", nên điểm rất cao. Cặp 2 chỉ chung **khái niệm ngầm**, còn bề mặt từ vựng thì gần như không giao nhau — "Sigma Gold" so với "sinh hoạt phí", "học bổng" so với "trợ cấp".
>
> Kết luận: embedding từ một mô hình nhỏ đa ngữ như MiniLM nắm được **cách diễn đạt khác nhau của cùng một ý** (cặp 1, cặp 4) nhưng chưa nắm được **suy luận phạm trù** (cặp 2). Nó nằm đâu đó giữa so khớp chuỗi thuần tuý và hiểu ngữ nghĩa thật sự.
>
> Phát hiện này khớp với kết quả ở Mục 5: TF-IDF từ vựng lại **thắng** MiniLM trên bộ câu hỏi của nhóm, vì các câu ấy giàu thực thể đặc trưng ("Vi mạch bán dẫn", "Đinh Thiện Lý", "Vallet") — đúng chỗ mà khớp từ khoá mạnh còn suy luận phạm trù không cần dùng đến.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Vai trò của tôi trong nhóm: **chạy đối chứng có kiểm soát**. Thay vì thử một chiến lược, tôi chạy **cả 3 chiến lược chunking trên cùng bộ 5 câu hỏi chung của nhóm**, lặp lại toàn bộ với **3 bộ nhúng** (MiniLM, TF-IDF, Gemini), rồi thêm một thí nghiệm nữa **đổi LLM trong khi giữ nguyên retrieval** — để tách riêng ảnh hưởng của chiến lược, của bộ nhúng, và của mô hình sinh câu trả lời.

Công cụ: `benchmark/run_comparison.py`, `benchmark/run_llm_ab.py` · Corpus: `data/scholarship/` (8 tài liệu) · Kết quả đầy đủ: `benchmark/RESULTS.md`

### Bảng so sánh 3 chiến lược × 2 bộ nhúng

| Chiến lược | #chunk | Dài TB | MiniLM | TF-IDF | Gemini |
|---|---|---|---|---|---|
| A. `FixedSizeChunker(300, 50)` | 208 | 296 | 80% / 20% | 100% / 60% | 80% / **60%** |
| B. `SentenceChunker(4)` | 83 | 619 | 80% / **40%** | 100% / **80%** | 80% / **60%** |
| C. `RecursiveChunker(350)` | 203 | 253 | 80% / **40%** | 100% / **80%** | **100%** / 40% |

*(mỗi ô: Hit@3 / Phủ đáp án)*

**Không chiến lược nào thắng ở cả ba backend.** TF-IDF xếp B và C đồng hạng nhất; Gemini lại xếp C cao nhất về Hit@3 nhưng **thấp nhất** về phủ đáp án; MiniLM thì B và C hoà. Thứ hạng đảo liên tục theo bộ nhúng — đây chính là lý do không thể trả lời "chiến lược nào tốt nhất" mà không nêu rõ chạy trên bộ nhúng nào.

### Thứ hạng tài liệu đúng trong top-3 (bộ nhúng MiniLM)

| Chiến lược | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| A. Fixed | 2 | — | 1 | 1 | 3 |
| B. Sentence | 1 | 1 | 1 | 1 | — |
| C. Recursive | 2 | 3 | 1 | 1 | — |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4 / 5** với cả ba chiến lược (MiniLM); **5 / 5** với TF-IDF.

### Ba phát hiện từ thí nghiệm của tôi

**1. Hit@3 không phân biệt được chiến lược — độ phủ đáp án thì có.**
Cả ba chiến lược đều đạt Hit@3 80% (MiniLM) và 100% (TF-IDF), nhìn như hoà nhau. Nhưng chỉ số tôi tự thêm — *đáp án có thực sự nằm trong ngữ cảnh top-3 không* — tách được rõ: A chỉ 20%, B và C 40%. Tìm đúng **tài liệu** là việc dễ; lấy đúng **mẩu chứa số liệu** mới khó. Nếu chỉ báo cáo Hit@3 thì hệ thống trông tốt hơn thực tế rất nhiều.

**2. Bẫy metadata của nhóm KHÔNG kích hoạt với bộ nhúng của tôi.**
Kết quả A/B trên Q5 với MiniLM: cả 3 chiến lược đều `[không đổi]`, số tài liệu sai đối tượng lọt top-3 = **0**. Lý do là MiniLM xếp `lstf` và `haui` lên trên, nên `hust-postgrad-research` (`audience=faculty`) không bao giờ vào top-3 để mà cần lọc. Trong khi đó với `bge-m3` (Vinh) và Gemini (Bá Quân), tài liệu faculty xếp **hạng 2** và filter thực sự cứu được kết quả.
Chạy lại với **Gemini embedding** thì bức tranh rõ hơn: bẫy **có nổ, nhưng chỉ với chiến lược Recursive** — `hust-postgrad-research` (`audience=faculty`) xếp **hạng 2** khi không lọc, và bật filter đẩy được tài liệu đúng lên. Hai chiến lược Fixed và Sentence vẫn không kích hoạt được.

**Kết luận:** một bẫy metadata chỉ "nổ" khi **cả bộ nhúng lẫn chiến lược chunking** cùng đưa tài liệu gây nhiễu lên đủ cao. Nói "filter có hiệu quả" mà không nêu rõ chạy trên cấu hình nào là chưa đủ — cùng một corpus, cùng một câu hỏi, mà 2 trong 3 chiến lược không thấy tác dụng gì.

**3. Với corpus này, TF-IDF từ vựng *thắng* embedding ngữ nghĩa nhỏ.**
TF-IDF đạt Hit@3 100% và phủ đáp án 60–80%, cao hơn MiniLM (80% / 20–40%) ở mọi chiến lược. Nguyên nhân: 5 câu hỏi của nhóm chứa các thực thể rất đặc trưng — "Vi mạch bán dẫn", "Đinh Thiện Lý", "Vallet". Khớp từ khoá giải quyết gọn, còn MiniLM (mô hình nhỏ, đa ngữ) lại kéo các tài liệu học bổng "nghe giống nhau" lên trên. Bài học: **không mặc định embedding ngữ nghĩa luôn tốt hơn** — phụ thuộc câu hỏi giàu thực thể hay giàu diễn giải.

**4. Đổi hẳn LLM, điểm không nhúc nhích — nút thắt là retrieval, không phải LLM.**
Để kiểm chứng giả thuyết "LLM là khâu yếu" (suy ra từ việc Vinh đạt top-3 5/5 nhưng chỉ 2/5 câu trả lời đạt ngưỡng), tôi cô lập biến số: **giữ nguyên hoàn toàn tầng truy xuất** (`RecursiveChunker(350)` + MiniLM, cùng ngữ cảnh top-3), chỉ thay mô hình sinh câu trả lời.

| Câu | Q1 | Q2 | Q3 | Q4 | Q5 | Tổng |
|---|---|---|---|---|---|---|
| Gemini 3.6 Flash | 1 | 1 | 1 | 2 | 0 | **5/10** |
| DeepSeek Chat | 1 | 1 | 1 | 2 | 0 | **5/10** |

Hai mô hình khác hãng, khác kiến trúc, khác hẳn quy mô — **điểm giống hệt nhau**. Cùng thắng ở Q4, cùng thua ở Q1/Q3/Q5, và cùng trả lời "Không có thông tin." ở đúng những chỗ ngữ cảnh thiếu đáp án. Nếu LLM là nút thắt thì thay LLM phải làm điểm xê dịch; nó không xê dịch.

Kết luận: **nút thắt nằm ở chunking**. Con số "top-3 hit 5/5" gây hiểu nhầm vì nó đo ở *mức tài liệu* — tài liệu đúng trong top-3 không đảm bảo *đoạn chứa đáp án* nằm trong đó. Đây đúng là lỗi "đúng tài liệu, sai mảnh" mà Giáp mô tả, tiếp cận từ một hướng hoàn toàn khác.
Công cụ: `benchmark/run_llm_ab.py`.

**Điều hay nhất tôi học được từ thành viên khác:**
> Phân tích lỗi của **Giáp** đi xa hơn tôi một bước. Tôi dừng ở chỗ "chunk cắt đứt đáp án"; Giáp chỉ ra nguyên nhân sâu hơn: khi một mục dài bị cắt thành nhiều mảnh, **các mảnh cùng mục có điểm gần bằng nhau**, nên mảnh nào lọt top-3 gần như ngẫu nhiên — và mảnh chứa con số cần tìm có thể bị một mục cùng chủ đề của tài liệu khác chen mất chỗ. Đây chính là lý do Q1 của tôi mất chuỗi `4.200.000` ở cả ba chiến lược.
>
> Giáp cũng đề xuất hướng sửa mà tôi chưa nghĩ tới: **small-to-big** — dùng chunk nhỏ (~200 ký tự) để *so khớp*, nhưng trả về cả mục cha để agent *đọc*. Tức là tách "đơn vị để tìm" khỏi "đơn vị để hiểu". Cách này sửa đúng cả hai lỗi mà chỉ số phủ đáp án của tôi phát hiện, và theo đối chứng của nhóm thì đạt 10/10 so với 6/10 của chiến lược heading.
>
> Từ **Hải** và **Bá Quân**, tôi học được giá trị của việc gắn lại tiêu đề cha vào chunk con. Từ **Vinh** và **Bá Quân**, tôi học cách chấm vế câu trả lời bằng ngưỡng tương đồng 0.80 thay vì chỉ xét top-k — đúng thứ mà benchmark của tôi còn thiếu vì chưa nối LLM.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá | Căn cứ |
|----------|-------------------|--------|
| Khởi động (Warm-up) | 5 / 5 | Giải thích đủ ý (góc vs độ lớn), ví dụ lấy từ chính corpus nhóm và có số đo thực tế kèm theo |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 | Giải thích được **lý do** thiết kế chứ không chỉ mô tả code: chọn tiền lọc thay vì hậu lọc (tránh `top_k` bị chunk sai đối tượng chiếm chỗ), gộp ngược mảnh vụn trong `RecursiveChunker`, nhúng một lần tái dùng nhiều truy vấn, đặt ngữ cảnh trước câu hỏi trong prompt để giảm ảo giác |
| Hoàn thiện code (Core Implementation) | 30 / 30 | 42/42 test pass |
| Dự đoán độ tương tự | 5 / 5 | Dự đoán đúng 4/5; phân tích được vì sao cặp 2 sai và nối được với kết quả ở Mục 5 |
| Kết quả truy xuất của tôi | 10 / 10 | Phép đo duy nhất trong nhóm cố định được biến số (3 chiến lược × 2 bộ nhúng, cùng bộ 5 câu chung); tự thiết kế chỉ số **phủ đáp án** ở mức chunk mà Hit@3 không đo được; ba phát hiện có số liệu chứng minh, trong đó phát hiện về bẫy metadata phụ thuộc bộ nhúng là kết quả không thành viên nào khác có |
| **Tổng phần cá nhân** | **60 / 60** | |
