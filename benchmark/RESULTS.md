# Kết quả benchmark — Nguyễn Quang Huy (2A202602421)

Bộ 5 câu chung của nhóm, 3 chiến lược chunking, **3 bộ nhúng** + **đối chứng 2 LLM**.
Corpus `data/scholarship/` (8 tài liệu).

## Tổng hợp

| Chiến lược | MiniLM | TF-IDF | Gemini |
|---|---|---|---|
| A. Fixed(300,50) | 80% / 20% | 100% / 60% | 80% / 60% |
| B. Sentence(4) | 80% / 40% | 100% / 80% | 80% / 60% |
| C. Recursive(350) | 80% / 40% | 100% / 80% | 100% / 40% |

*(Hit@3 / Phủ đáp án)* — không chiến lược nào thắng ở cả ba backend.

## Đối chứng LLM (giữ nguyên retrieval)

```
========================================================================================
ĐỐI CHỨNG LLM — cùng retrieval, khác mô hình sinh
========================================================================================
Retrieval cố định : recursive {'chunk_size': 350} + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (203 chunk)
LLM Gemini        : gemini:gemini-3.6-flash
LLM DeepSeek      : deepseek:deepseek-chat
Ngưỡng đạt        : tương đồng >= 0.8

[Q1] Mức học bổng dành cho sinh viên trúng tuyển ngành Vi mạch bán dẫn tại Đại 
   tài liệu đúng hạng 2  |  gold: 4.200.000 đồng/tháng.
   Gemini   : sim=0.0612 TRƯỢT [1/2]  Không có thông tin.
   DeepSeek : sim=0.0612 TRƯỢT [1/2]  Không có thông tin.

[Q2] Để duy trì học bổng Đinh Thiện Lý, sinh viên cần đạt kết quả học tập (CGPA
   tài liệu đúng hạng 3  |  gold: CGPA từ 8.0 trở lên theo thang điểm 10.
   Gemini   : sim=0.0731 TRƯỢT [1/2]  Không có thông tin.
   DeepSeek : sim=0.5676 TRƯỢT [1/2]  8.0 trở lên.

[Q3] Để Quỹ Đinh Thiện Lý thực hiện giải ngân thanh toán học bổng, phía nhà trư
   tài liệu đúng hạng 1  |  gold: (1) Bộ hồ sơ sinh viên, (2) Tổng số tiền học bổng, (3) Thông tin tài k
   Gemini   : sim=0.0722 TRƯỢT [1/2]  Không có thông tin.
   DeepSeek : sim=0.0755 TRƯỢT [1/2]  Context không nêu cụ thể các thông tin cần cung cấp.

[Q4] Hãy liệt kê các mức phần trăm hỗ trợ trong chính sách miễn, giảm học phí t
   tài liệu đúng hạng 1  |  gold: 3 mức: 100%, 70% và 50% học phí.
   Gemini   : sim=0.9152 ĐẠT  [2/2]  100%, 70% và 50% học phí.
   DeepSeek : sim=0.9152 ĐẠT  [2/2]  100%, 70% và 50% học phí.

[Q5] Tiêu chuẩn về kết quả học tập để sinh viên được nhận hỗ trợ 100% học phí l
   tài liệu đúng hạng -  |  gold: Không có tiêu chuẩn về kết quả học tập — chỉ cần sinh viên thuộc diện 
   Gemini   : sim=0.1886 TRƯỢT [0/2]  Không có thông tin.
   DeepSeek : sim=0.1886 TRƯỢT [0/2]  Không có thông tin.

========================================================================================
BẢNG ĐIỂM — retrieval giống hệt nhau, khác biệt duy nhất là LLM
   Gemini    : 5/10
   DeepSeek  : 5/10
========================================================================================
```

## Phụ lục — Gemini embedding

```
Bộ nhúng: gemini-embedding-001
Corpus:   8 tài liệu
======================================================================================
Chiến lược                          #chunk   dài TB   Hit@3    MRR   Phủ đáp án
======================================================================================
A. Fixed — fixed              208      296    80%   0.80         60%
B. Sentence — sentence            83      619    80%   0.70         60%
C. Recursive — recursive          203      253   100%   0.87         40%
Thứ hạng tài liệu đúng trong top-3 (- = trượt):
                                      Q1    Q2    Q3    Q4    Q5
A. Fixed                               1     1     1     1     -
B. Sentence                            1     1     1     2     -
C. Recursive                           1     1     1     1     3
Mẩu số liệu BỊ MẤT khỏi ngữ cảnh top-3 (chunk cắt đứt đáp án):
  A. Fixed          {'Q3': ['Tổng số tiền học bổng', 'tài khoản ngân hàng'], 'Q5': ['100%, 70% và 50%']}
  B. Sentence       {'Q1': ['4.200.000'], 'Q5': ['100%, 70% và 50%']}
  C. Recursive      {'Q1': ['4.200.000'], 'Q3': ['Tổng số tiền học bổng', 'tài khoản ngân hàng'], 'Q5': ['100%, 70% và 50%']}
======================================================================================
A/B metadata filter — Q5: Tiêu chuẩn về kết quả học tập để sinh viên được nhận hỗ trợ 100% học phí là gì?
======================================================================================
A. Fixed  [không đổi]  — tài liệu sai đối tượng lọt top-3: 0
   không lọc  : ['haui-financial-aid-scholarships', 'haui-financial-aid-scholarships', 'lstf-dinh-thien-ly-scholarship']
   lọc student: ['haui-financial-aid-scholarships', 'haui-financial-aid-scholarships', 'lstf-dinh-thien-ly-scholarship']
B. Sentence  [không đổi]  — tài liệu sai đối tượng lọt top-3: 0
   không lọc  : ['haui-financial-aid-scholarships', 'lstf-dinh-thien-ly-scholarship', 'lstf-dinh-thien-ly-scholarship']
   lọc student: ['haui-financial-aid-scholarships', 'lstf-dinh-thien-ly-scholarship', 'lstf-dinh-thien-ly-scholarship']
C. Recursive  [ĐỔI KẾT QUẢ]  — tài liệu sai đối tượng lọt top-3: 1
   không lọc  : ['haui-financial-aid-scholarships', 'hust-postgrad-research-scholarships', 'hust-financial-aid-for-students']
   lọc student: ['haui-financial-aid-scholarships', 'hust-financial-aid-for-students', 'haui-financial-aid-scholarships']
Gold: Không có tiêu chuẩn về kết quả học tập — chỉ cần sinh viên thuộc diện chính sách theo quy định.
Bẫy:  hust-postgrad-research-scholarships (audience=faculty) cũng nói về 'học bổng toàn phần bằng 100% học phí' nhưng điều kiện khác hẳn (CPA từ 3.2, hoặc bài báo ISI/Scopus). Câu hỏi không nêu người hỏi là ai nên không lọc sẽ lẫn hai tài liệu và trả lời sai đối tượng.
[exited with code 0]
```
