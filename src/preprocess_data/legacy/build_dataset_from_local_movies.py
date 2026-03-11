"""
Master Script: Build Dataset from Local Full Movies

This script allows you to take raw, full-length movie files (.mp4, .mkv, .avi)
that you already have locally, and automatically process them into a structured
Visual Dataset matching the exact MovieNet format (Keyframes).

Requirements:
1. `ffmpeg` must be installed and accessible in your system PATH.
2. Put all your raw movie files in a single folder (e.g., `D:\Raw_Movies`).
3. Rename the movie files to match their IMDb ID exactly (e.g., `tt0120338.mp4` for Titanic).
"""

import os
import subprocess
from pathlib import Path
import time

# ================= Configuration =================
# Thư mục chứa các file phim gốc của bạn (bạn tự tải về bỏ vào đây)
RAW_MOVIES_DIR = r"D:\Study\School\project_ky4\data\Raw_Movies"

# Thư mục đầu ra chứa Keyframes (Chuẩn cấu trúc MovieNet)
OUTPUT_KEYFRAMES_DIR = r"D:\Study\School\project_ky4\data\Standalone_Dataset\shot_keyf"

# Tuỳ chỉnh tần suất trích xuất khung hình (thường MovieNet lấy 3s = 1 hình)
FPS_EXTRACT = "1/3"
# =================================================


def setup_directories():
    Path(RAW_MOVIES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_KEYFRAMES_DIR).mkdir(parents=True, exist_ok=True)
    print(f" Thư mục phim gốc: {RAW_MOVIES_DIR}")
    print(f" Thư mục Keyframes đầu ra: {OUTPUT_KEYFRAMES_DIR}")
    print("-" * 50)


def extract_keyframes_from_movie(movie_file, movie_id):
    """Sử dụng FFmpeg để băm nhỏ file phim thành các Keyframes liên tục."""

    movie_out_dir = os.path.join(OUTPUT_KEYFRAMES_DIR, movie_id)
    Path(movie_out_dir).mkdir(parents=True, exist_ok=True)

    # Kiểm tra xem đã xử lý chưa
    existing_frames = list(Path(movie_out_dir).glob("*.jpg"))
    if len(existing_frames) > 10:
        print(
            f" Phim [{movie_id}] có vẻ đã được xử lý ({len(existing_frames)} hình). Bỏ qua."
        )
        return

    print(f" Đang băm phim [{movie_id}] thành Keyframes (Cứ mỗi 3s lấy 1 hình)...")
    print("   Việc này có thể mất từ 5-10 phút tuỳ độ dài phim và cấu hình máy.")

    start_time = time.time()

    # Lệnh FFmpeg để trích xuất hình ảnh
    command = [
        "ffmpeg",
        "-i",
        str(movie_file),
        "-vf",
        f"fps={FPS_EXTRACT},scale=-1:360",  # Resize chiều cao về 360p để tiết kiệm dung lượng, giữ nguyên tỷ lệ
        "-qscale:v",
        "3",  # Chất lượng ảnh (1-31, số nhỏ = chất lượng cao, 3 là khá tốt và nhẹ)
        os.path.join(movie_out_dir, "shot_%04d_img_0.jpg"),
        "-v",
        "warning",  # Chỉ in ra cảnh báo hoặc lỗi để đỡ rác console
        "-y",
    ]

    try:
        subprocess.run(command, check=True)
        elapsed = time.time() - start_time
        frame_count = len(list(Path(movie_out_dir).glob("*.jpg")))
        print(
            f" Xong [{movie_id}] trong {elapsed:.1f}s. Trích xuất thành công {frame_count} hình ảnh."
        )

    except subprocess.CalledProcessError as e:
        print(f" LỖI trong quá trình xử lý phim [{movie_id}]: {e}")
    except FileNotFoundError:
        print(
            " LỖI: Không tìm thấy 'ffmpeg'. Bạn phải cài phần mềm FFmpeg vào máy trước nhé."
        )


def main():
    print("=========================================================")
    print("    BỘ XỬ LÝ PHIM THÔ THÀNH MOVIENET DATASET ")
    print("=========================================================\n")

    setup_directories()

    if not os.path.exists(RAW_MOVIES_DIR):
        print("Không tìm thấy thư mục phim gốc. Hãy tạo và copy file mp4 vào.")
        return

    # Lấy danh sách các file video
    valid_extensions = {".mp4", ".mkv", ".avi", ".mov"}
    movie_files = [
        f
        for f in os.listdir(RAW_MOVIES_DIR)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not movie_files:
        print(f"️ Thư mục '{RAW_MOVIES_DIR}' đang trống.")
        print(" HƯỚNG DẪN:")
        print("1. Hãy tải các tập phim bạn cần về máy (ví dụ Titanic.mp4).")
        print(
            "2. Đổi tên file đó thành IMDb ID tương ứng (VD: đổi thành tt0120338.mp4)."
        )
        print("3. Bỏ vào thư mục Raw_Movies và chạy lại script này.")
        return

    print(f"Phát hiện {len(movie_files)} tệp tin phim. Bắt đầu xử lý...\n")

    for i, file_name in enumerate(movie_files, 1):
        movie_path = os.path.join(RAW_MOVIES_DIR, file_name)
        movie_id = os.path.splitext(file_name)[0]  # e.g. "tt0120338"

        print(f"--- Tiến trình: {i}/{len(movie_files)} ---")
        extract_keyframes_from_movie(movie_path, movie_id)

    print("\n HOÀN TẤT TOÀN BỘ!")
    print(
        f"Toàn bộ Dataset Hình Ảnh của bạn hiện nằm gọn gàng tại: {OUTPUT_KEYFRAMES_DIR}"
    )
    print(
        "Bây giờ bạn có thể trỏ model RAG của bạn vào thư mục Keyframes này để tìm kiếm!"
    )


if __name__ == "__main__":
    main()
