"""Bộ 5 benchmark query CHUNG của nhóm K4-L3A + gold answer trích từ data/scholarship/.

Bộ này thống nhất với bài làm của Khánh, Vinh và Bá Quân, để kết quả của
các thành viên so sánh được với nhau.
"""

QUERIES = [
    {
        "id": "Q1",
        "kind": "tra_so_lieu",
        "question": "Mức học bổng dành cho sinh viên trúng tuyển ngành Vi mạch bán dẫn "
                    "tại Đại học Công nghiệp Hà Nội là bao nhiêu tiền một tháng?",
        "gold_doc": "haui-financial-aid-scholarships",
        "gold_answer": "4.200.000 đồng/tháng.",
        "gold_keys": ["4.200.000"],
        "needs_filter": None,
    },
    {
        "id": "Q2",
        "kind": "hoi_dieu_kien",
        "question": "Để duy trì học bổng Đinh Thiện Lý, sinh viên cần đạt kết quả học tập "
                    "(CGPA) tối thiểu là bao nhiêu nếu tính theo thang điểm 10?",
        "gold_doc": "lstf-dinh-thien-ly-scholarship",
        "gold_answer": "CGPA từ 8.0 trở lên theo thang điểm 10.",
        "gold_keys": ["8.0 trở lên"],
        "needs_filter": None,
    },
    {
        "id": "Q3",
        "kind": "hoi_quy_trinh",
        "question": "Để Quỹ Đinh Thiện Lý thực hiện giải ngân thanh toán học bổng, "
                    "phía nhà trường cần gửi văn bản với những thông tin gì?",
        "gold_doc": "lstf-dinh-thien-ly-scholarship",
        "gold_answer": "(1) Bộ hồ sơ sinh viên, (2) Tổng số tiền học bổng, "
                       "(3) Thông tin tài khoản ngân hàng.",
        "gold_keys": ["Tổng số tiền học bổng", "tài khoản ngân hàng"],
        "needs_filter": None,
    },
    {
        "id": "Q4",
        "kind": "liet_ke",
        "question": "Hãy liệt kê các mức phần trăm hỗ trợ trong chính sách miễn, giảm "
                    "học phí theo Nghị định của Chính phủ đối với sinh viên?",
        "gold_doc": "hust-financial-aid-for-students",
        "gold_answer": "3 mức: 100%, 70% và 50% học phí.",
        "gold_keys": ["100%, 70% và 50%"],
        "needs_filter": None,
    },
    {
        "id": "Q5",
        "kind": "metadata_trap",
        "question": "Tiêu chuẩn về kết quả học tập để sinh viên được nhận hỗ trợ "
                    "100% học phí là gì?",
        "gold_doc": "hust-financial-aid-for-students",
        "gold_answer": "Không có tiêu chuẩn về kết quả học tập — chỉ cần sinh viên "
                       "thuộc diện chính sách theo quy định.",
        "gold_keys": ["100%, 70% và 50%"],
        "needs_filter": {"audience": "student"},
        "distractor_doc": "hust-postgrad-research-scholarships",
        "why_filter": "hust-postgrad-research-scholarships (audience=faculty) cũng nói về "
                      "'học bổng toàn phần bằng 100% học phí' nhưng điều kiện khác hẳn "
                      "(CPA từ 3.2, hoặc bài báo ISI/Scopus). Câu hỏi không nêu người hỏi "
                      "là ai nên không lọc sẽ lẫn hai tài liệu và trả lời sai đối tượng.",
    },
]

STRATEGIES = {
    "A": {"owner": "Fixed", "chunker": "fixed",
          "params": {"chunk_size": 300, "overlap": 50},
          "note": "Cắt theo độ dài cố định, cửa sổ trượt 50 ký tự."},
    "B": {"owner": "Sentence", "chunker": "sentence",
          "params": {"max_sentences_per_chunk": 4},
          "note": "Cắt theo ranh giới câu, 4 câu mỗi chunk."},
    "C": {"owner": "Recursive", "chunker": "recursive",
          "params": {"chunk_size": 350},
          "note": "Cắt đệ quy theo đoạn > dòng > câu > từ, trần 350 ký tự."},
}
