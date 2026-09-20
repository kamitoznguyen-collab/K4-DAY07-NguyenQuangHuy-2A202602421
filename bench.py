import json
import glob
import os
import yaml
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.models import Document

def main():
    print("Khởi tạo Chunker và Store...")
    chunker = RecursiveChunker(chunk_size=1000)
    store = EmbeddingStore()

    print("Đọc dữ liệu từ data/...")
    md_files = glob.glob("data/**/*.md", recursive=True)
    all_chunks = []
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        metadata = {"source": file_path}
        
        # Parse YAML frontmatter if present
        if content.startswith("---"):
            try:
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    body = parts[2].strip()
                    parsed_meta = yaml.safe_load(frontmatter)
                    if isinstance(parsed_meta, dict):
                        metadata.update(parsed_meta)
                    content = body
            except Exception as e:
                print(f"Warning: Failed to parse YAML in {file_path}: {e}")
                
        # Use chunker
        chunks = chunker.chunk(content)
        for i, text in enumerate(chunks):
            # Create a document for each chunk
            doc_id = f"{os.path.basename(file_path)}_chunk_{i}"
            all_chunks.append(Document(id=doc_id, content=text, metadata=metadata))
            
    print(f"Đã tạo {len(all_chunks)} chunks từ {len(md_files)} file.")
    
    print("Nạp chunks vào Vector Store...")
    store.add_documents(all_chunks)
    print(f"Số chunk đã nạp: {store.get_collection_size()}")
    
    print("Đọc benchmark_queries.json...")
    with open("benchmark_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print("Chạy Benchmark...")
    results_str = f"Kết quả Benchmark\nSố lượng file: {len(md_files)}\nSố lượng chunk đã nạp: {store.get_collection_size()}\n"
    results_str += "=" * 50 + "\n\n"
    
    for q in queries:
        query_text = q["query"]
        gold = q["gold_answer"]
        meta_filter = q.get("metadata_filter", {})
        
        print(f"\n[Câu hỏi]: {query_text}")
        print(f"[Filter]: {meta_filter}")
        print(f"[Gold Answer]: {gold}")
        
        results_str += f"[Câu hỏi]: {query_text}\n"
        results_str += f"[Filter]: {meta_filter}\n"
        results_str += f"[Gold Answer]: {gold}\n"
        
        # Search
        results = store.search_with_filter(query_text, top_k=3, metadata_filter=meta_filter)
        
        for i, res in enumerate(results):
            score = res.get("score", 0.0)
            source = res["metadata"].get("source", "Unknown")
            doc_id = res["id"]
            content = res["content"][:150].replace('\n', ' ') + "..."
            
            out_line = f"  - Top {i+1} [Score: {score:.3f}] - {doc_id}\n    Nội dung: {content}"
            print(out_line)
            results_str += out_line + "\n"
        
        results_str += "-" * 50 + "\n"
        
    with open("ket_qua_benchmark.txt", "w", encoding="utf-8") as f:
        f.write(results_str)
        
    print("\n✅ Đã lưu kết quả chi tiết vào ket_qua_benchmark.txt")

if __name__ == "__main__":
    main()
