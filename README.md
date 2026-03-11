# 🎬 MovieRAG "Super System"

MovieRAG là một hệ thống **Agentic Video Retrieval-Augmented Generation (VideoRAG)** đa luồng, được thiết kế để tìm kiếm tình tiết phim, đoạn trích, thông tin kiến thức sự kiện và nhân vật trong phim một cách chính xác dựa trên đa phương thức (văn bản, hình ảnh, video). Hệ thống là sự kết hợp của kiến trúc GraphRAG, Vector RAG và năng lực lý luận (reasoning) tự động từ các mô hình VLM tiên tiến (Groq/Gemma/Llama3).

---

## 📌 Tính năng Nổi bật

- 🧠 **Agentic Workflow**: Tự động phân tích ngữ cảnh ưu tiên (Contextualize), phân loại ý định (Route Intent: Visual, Knowledge, Multimodal, Chat, Dialog), và thực hiện trích xuất nhiều lần để có câu trả lời tốt nhất.
- 🖼️ **Truy vấn Đa Phương Thức (Multi-modal Retrieval)**: Hỗ trợ tìm kiếm theo câu hỏi văn bản, tải ảnh lên hoặc tải video lên để tìm lại các cảnh quay tương ứng trong cơ sở dữ liệu.
- 🕷️ **Graph-driven Textual Grounding**: Sử dụng đồ thị tri thức (Knowledge Graph - Neo4j) kết hợp với FAISS Vector để xây dựng ngữ cảnh sâu cho phim.
- ⚙️ **Auto-Ingestion Pipeline (8 bước)**: Tự động hóa hoàn toàn quá trình tải video mới vào hệ thống (từ phát hiện cảnh, bóc băng phụ đề bằng STT, đến phân tích cảnh quay bằng VLM và đưa vào CSDL).
- 🎨 **Giao diện Cao cấp (Premium UI)**: Tương tác qua Gradio với giao diện Dark Mode, hiển thị Evidence minh bạch (Keyframes, Video clips, và suy luận logic của Agent).

---

## 🏗️ Kiến trúc Hệ thống

Hệ thống được chia làm hai phần chính:
1. **Pipeline Tiền xử lý Dữ liệu (`preprocess_data`)**: Chuyển đổi video thô thành các vector nhúng (embeddings) và đồ thị.
2. **Công cụ RAG & UI (`movierag`)**: Gateway tương tác với người dùng.

### Các loại Index đang sử dụng:
- **Visual / Hybrid FAISS Index**: Lưu trữ vector của hàng chục ngàn keyframe được trích xuất từ video thông qua mô hình mã hóa CLIP. Hỗ trợ tìm kiếm hình ảnh bằng văn bản (Zero-shot) và tìm kiếm hình ảnh bằng hình ảnh.
- **Knowledge Text FAISS Index**: Lưu trữ metadata, kịch bản (scripts) và các tóm tắt phân tích cảnh quay.
- **Dialogue Indexer**: Lưu trữ và tìm kiếm theo sát từng mốc thời gian phụ đề (SRT).
- **Knowledge Graph (Neo4j)**: Lưu trữ các Entity (Nhân vật, Địa điểm, Sự kiện) và Relation (Mối quan hệ) để hỗ trợ multi-hop reasoning (lý luận đa bước).

---

## 📂 Tổ chức Mã nguồn

```text
VideoRag/
├── data/                       # Chứa CSDL, Indexes, Annotation (Tải từ Google Drive)
├── docker-compose.neo4j.yml    # Cấu hình khởi tạo Neo4j Graph Database
├── scripts/                    # Các script hỗ trợ bổ sung
├── src/
│   ├── movierag/               # CORE SOURCE: Pipeline RAG và Website interface
│   │   ├── app.py              # Giao diện web Gradio
│   │   ├── main.py             # Entry point (build, demo, verify)
│   │   └── pipeline/           # Logic Agentic Workflow đa luồng
│   ├── preprocess_data/        # MODULE: Xử lý video 8 bước tự động
│   │   ├── ingest_movie.py     # Script chính ingest video mới
│   │   └── video/              # Các tool Auto-crop, Extract frames, STT
├── requirements.txt            # Thư viện phụ thuộc
├── Task.md                     # Bảng kế hoạch chi tiết các Phase (Phase 1-4)
└── README.md                   # Tài liệu bạn đang xem
```

---

## ⚙️ Cài đặt & Chuẩn bị Dữ liệu

### 1. Dữ liệu (Google Drive)
Toàn bộ dữ liệu của dự án có kích thước lớn (video, FAISS index, CSDL đồ thị) sẽ **không được commit lên Github**. 
👉 **Bạn cần tải thư mục `data` từ Google Drive (theo link đính kèm nội bộ)** và giải nén vào thư mục gốc của dự án `VideoRag/data/`.
*(Lưu ý: Bạn cũng có thể dùng `movie_data_subset_20` nếu đang test ở môi trường thu nhỏ).*

### 2. Cài đặt Môi trường (Environment Setup)
Yêu cầu hệ thống phải có Python >= 3.10. Khuyến khích sử dụng `conda` hoặc Python `venv` để quản lý độc lập các thư viện.
```bash
# Tạo môi trường conda
conda create -n videorag python=3.10 -y
conda activate videorag

# Cài đặt toàn bộ các thư viện cần thiết
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường (`.env`)
Tạo file `.env` ở thư mục gốc chứa các khóa truy cập API. Toàn bộ tính năng LLM/VLM hiện đang sử dụng qua Groq API:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Khởi động Graph Database (Neo4j)
Hệ thống sử dụng Docker để chạy Neo4j cục bộ trên cổng `7474` và `7687`:
```bash
docker-compose -f docker-compose.neo4j.yml up -d
```
*(Lưu ý: Mật khẩu mặc định khi hệ thống truy cập DB là `movierag123`)*

---

## 🚀 Hướng dẫn Sử dụng (Usage Guide)

Dự án cung cấp CLI mạnh mẽ cho hầu hết các thao tác được xây dựng thông qua module `movierag.main` và `preprocess_data.__main__`.

### A. Tích hợp Phim Mới vào CSDL (Auto-Ingestion)
Để đưa một đoạn video/phim hoàn toàn mới vào CSDL, chạy lệnh sau:
```bash
python -m preprocess_data ingest đường_dẫn_đến_video/phim.mp4 --id mã_phim_imdb_hoặc_tên
# Ví dụ:
# python -m preprocess_data ingest my_videos/inception.mp4 --id tt1375666
```

### B. Cập nhật & Xây dựng Index lại (Rebuild)
Sau khi thay đổi dữ liệu hoặc code nhúng, bạn có thể thiết lập chỉ mục FAISS lại từ đầu:
```bash
# Dùng lệnh build của movierag
python -m movierag.main build --data-dir <đường dẫn thư mục dataset>
```

### C. Khởi chạy Giao diện Trợ lý Tìm kiếm RAG (Gradio App)
Mở giao diện tương tác người dùng, truy vấn bằng text hoặc ảnh:
```bash
python -m movierag.main demo
```
👉 *Mở trình duyệt tại địa chỉ:* `http://127.0.0.1:7861`

### D. Kiểm thử & Đánh giá Chất lượng (Verify & Evaluate)
- **Verify test nhanh**: Đảm bảo pipeline hiện tại không gặp lỗi trong môi trường thực thi.
    ```bash
    python -m movierag.main verify
    ```
- **Evaluation Database (Sử dụng Groq)**: Chấm điểm độ sinh xác của câu trả lời tự động bằng LLM-as-a-judge.
    ```bash
    python -m movierag.main eval --dataset data/eval_queries.json
    ```

---

## 👥 Nhóm Phát triển
Dự án được xây dựng và chia giai đoạn công việc dựa trên thiết kế (Xem chi tiết tại `Task.md`):
- 🔵 **Sơn (Lead)**: Thiết kế Core Architecture, Workflow Agentic RAG đa luồng, xử lý dữ liệu FAISS index, và triển khai Gradio UI cao cấp.
- 🟠 **Hiếu**: Xây dựng Visual Retrieval Pipeline (nhúng CLIP, trích xuất Keyframe, so khớp ảnh/video) và thử nghiệm các mô hình VLM.
- 🟢 **Thắng**: Tìm kiếm tài liệu, tích hợp module nhận diện và xử lý phụ đề (Whisper STT), xây dựng Context Builder, và đánh giá.
- 🟣 **Vinh**: Thu thập Metadata API (IMDB), tải Dataset thô, thiết lập dự án ban đầu, cấu hình Neo4j Graph DB và 최 ưu hóa truy vấn thông qua endpoints.