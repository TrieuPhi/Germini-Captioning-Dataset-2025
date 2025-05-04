# 🚦 Vietnamese Image Captioning Project
[🌟 VietNamese](README.md) | [🌏 English](README_en.md)
Dự án này cung cấp giải pháp tạo caption tự động bằng tiếng Việt cho hình ảnh sử dụng mô hình học sâu từ Hugging Face và được triển khai thông qua Flask API.

| **notebook** | **open in colab / kaggle** | **complementary materials** | **repository / paper** |
|:------------:|:-------------------------------------------------:|:---------------------------:|:----------------------:|
| [LLM_Expose_Flask_API_With_Ngrok](https://github.com/TrieuPhi/Huggingface-Captioning-Data/blob/main/LLM_Ngok_API.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/TrieuPhi/Huggingface-Captioning-Data/blob/main/LLM_Ngok_API.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/trieuphi/llm-ngrok-api-using-kaggle)  |   | [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/TrieuPhi/Huggingface-Captioning-Data/tree/main) |

> **Tạo mô tả tự động cho hình ảnh giao thông bằng tiếng Việt, ứng dụng AI và tăng cường dữ liệu hiện đại.**

---

## 📚 Tổng quan

Dự án này cung cấp giải pháp xây dựng bộ dữ liệu hình ảnh giao thông kèm caption ngắn gọn bằng tiếng Việt, phục vụ các bài toán thị giác máy tính và hỗ trợ người khiếm thị. Toàn bộ pipeline được tự động hóa từ thu thập, làm sạch, sinh caption đến tăng cường dữ liệu.

---

## 🗂️ Cấu trúc thư mục

```
Germini-Captioning-Dataset-2025/
├── 1.crawl_data/                # Thu thập dữ liệu hình ảnh từ Google Images (SerpApi)
│   ├── output/                  # Lưu metadata, file CSV kết quả
│   └── python/                  # Script crawl dữ liệu
├── 2.data_preprocessing/        # Làm sạch và xử lý dữ liệu
│   └── jupyter/                 # Notebook xử lý
├── 3.labels_short_captions/     # Sinh caption ngắn cho ảnh bằng Gemini API
│   └── python/                  # Script sinh caption
├── 4.Image_data_augument/       # Tăng cường dữ liệu ảnh (augmentation)
│   └── python/                  # Script augmentation
├── image.png                    # Sơ đồ workflow
├── README.md                    # Tài liệu này
└── ...
```

---

## 🚀 Quy trình tổng thể

![Workflow](image.png)

1. **Crawl Data**  
   Thu thập ảnh giao thông từ Google Images qua SerpApi, lưu metadata và ảnh về máy.

2. **Data Preprocessing**  
   Làm sạch dữ liệu: loại bỏ URL hỏng, xử lý null, chuẩn hóa thông tin.

3. **Labels Captions**  
   Sử dụng API Gemini 2.0 Flash để sinh caption ngắn gọn (10-15 từ) cho từng ảnh.

4. **Image Augmentation**  
   Tăng cường dữ liệu bằng các kỹ thuật biến đổi ảnh hiện đại với [Albumentations](https://albumentations.ai/).

---

## 🏗️ Chi tiết từng bước

### 1️⃣ Crawl Data

- **Mục tiêu:** Thu thập ảnh giao thông đa dạng từ nhiều từ khóa (ví dụ: "đèn giao thông", "người đi bộ qua đường", "giao thông ban đêm", ...).
- **Công cụ:** [SerpApi](https://serpapi.com/) + Python script ([traffic_raw.py](1.crawl_data/python/traffic_raw.py))
- **Kết quả:**  
  - File `metadata.json` chứa thông tin chi tiết từng ảnh.
  - File `traffic_images_dataset.csv` tổng hợp URL, tiêu đề, nguồn, độ phân giải, ...
- **Ví dụ từ khóa:**  
  - "đèn tín hiệu giao thông cho người đi bộ", "giao thông công cộng hiện đại", "xe máy đậu trên vỉa hè", ...

### 2️⃣ Data Preprocessing

- **Mục tiêu:** Làm sạch dữ liệu, loại bỏ ảnh lỗi, chuẩn hóa các trường thông tin.
- **Công việc:**  
  - Loại bỏ URL không hợp lệ, ảnh không tải được.
  - Xử lý giá trị thiếu/null.
  - Giữ lại các trường cần thiết: `original_url`, `title`, `resolution`, ...
- **Kết quả:**  
  - File CSV sạch, sẵn sàng cho bước sinh caption.

### 3️⃣ Labels Short Captions

- **Mục tiêu:** Sinh caption ngắn gọn, súc tích cho từng ảnh.
- **Công cụ:**  
  - API Gemini 2.0 Flash ([label_short_captions.py](3.labels_short_captions/python/label_short_captions.py))
- **Quy trình:**  
  - Đọc file CSV đã làm sạch.
  - Gửi từng URL ảnh lên Gemini API với prompt tối ưu:
    > "Mô tả tổng quan nhất về nội dung trong tấm hình, tập trung vào tình hình giao thông hiện tại. Hãy giữ mô tả ngắn gọn (khoảng 10-15 từ trong 1 câu), sao cho cả câu mô tả không được quá 15 từ, để người mù có thể nắm bắt được thông tin nhanh chóng."
  - Lưu kết quả vào cột `short_caption`.
- **Kết quả:**  
  - File CSV chứa caption cho từng ảnh.

### 4️⃣ Image Augmentation

- **Mục tiêu:** Tăng số lượng và đa dạng bộ dữ liệu ảnh.
- **Công cụ:**  
  - [Albumentations](https://albumentations.ai/) ([data_augument.py](4.Image_data_augument/python/data_augument.py))
- **Kỹ thuật:**  
  - Điều chỉnh cường độ pixel (sáng/tối, tương phản, màu sắc)
  - Biến đổi không gian (xoay, phóng to/thu nhỏ, dịch chuyển)
  - Thêm nhiễu, làm mờ, cắt ngẫu nhiên, thay đổi kích thước
- **Kết quả:**  
  - Ảnh augmented lưu tại `augmented/images`
  - File CSV cập nhật caption cho ảnh augmented

---

## 💻 Hướng dẫn sử dụng nhanh

### 1. Cài đặt môi trường

```bash
cd Germini-Captioning-Dataset-2025/1.crawl_data/
conda env create -f env_crawl_data.yaml
conda activate crawl_data
pip install -r ../3.labels_short_captions/python/requirements.txt
pip install -r ../4.Image_data_augument/python/requirements.txt
```

### 2. Chạy từng bước pipeline

**Bước 1: Crawl dữ liệu**
```bash
python 1.crawl_data/python/traffic_raw.py
```

**Bước 2: Làm sạch dữ liệu**
- Sử dụng notebook hoặc script trong `2.data_preprocessing/jupyter/`

**Bước 3: Sinh caption**
```bash
python 3.labels_short_captions/python/label_short_captions.py
```

**Bước 4: Augmentation**
```bash
python 4.Image_data_augument/python/data_augument.py
```

---

## 📊 Ví dụ kết quả

| original_url | short_caption | augmented_image_path |
|--------------|--------------|---------------------|
| ...          | "Đường phố đông đúc xe cộ giờ cao điểm." | augmented/images/xxx.jpg |
| ...          | "Người đi bộ băng qua vạch kẻ đường."    | augmented/images/yyy.jpg |

---

## 📝 Đóng góp & Liên hệ

- Đóng góp ý tưởng, báo lỗi hoặc pull request tại [GitHub repository](https://github.com/TrieuPhi/Germini-Captioning-Dataset-2025/tree/main)
- Liên hệ: dtptrieuphidtp@gmail.com

---

## 📄 Giấy phép

Dự án sử dụng giấy phép [MIT License](LICENSE).

---

> **Made with ❤️ by Vietnamese AI Community**