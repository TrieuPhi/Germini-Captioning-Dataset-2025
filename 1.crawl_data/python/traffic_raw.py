# -*- coding: utf-8 -*-
from serpapi import GoogleSearch
import pandas as pd
import time
from tqdm import tqdm
import requests
import os
import json
from urllib.parse import urlparse, unquote
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import imghdr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../output/crawler.log'),
        logging.StreamHandler()
    ]
)

# Tạo thư mục images nếu chưa có
IMAGES_DIR = "../outputimages"
os.makedirs(IMAGES_DIR, exist_ok=True)

def clean_filename(filename):
    # Xử lý tên file, loại bỏ ký tự đặc biệt
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename[:200]  # Giới hạn độ dài tên file

def download_image(url, search_query):
    try:
        session = create_session_with_retries()
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            # Kiểm tra Content-Type
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                return None
                
            # Kiểm tra thời gian tồn tại của ảnh qua Last-Modified header
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                try:
                    modified_time = time.mktime(time.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT'))
                    current_time = time.time()
                    # Bỏ qua ảnh cũ hơn 5 năm
                    if (current_time - modified_time) > (5 * 365 * 24 * 60 * 60):
                        return None
                except:
                    pass
                
            # Tạo tên file từ URL và query
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1]
            if not file_extension:
                file_extension = '.jpg'
            
            # Tạo tên file an toàn
            base_filename = f"{clean_filename(search_query)}_{int(time.time())}{file_extension}"
            file_path = os.path.join(IMAGES_DIR, base_filename)
            
            # Lưu file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
    except Exception as e:
        print(f"Lỗi tải ảnh: {str(e)}")
        return None

def filter_image(image_data):
    # Kiểm tra kích thước tối thiểu
    min_width = 800
    min_height = 600
    width = image_data.get('original_width', 0)
    height = image_data.get('original_height', 0)
    
    if width < min_width or height < min_height:
        return False
        
    # Kiểm tra tỷ lệ khung hình
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
        
    return True

def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=3,
                   backoff_factor=0.5,
                   status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

search_queries = [
    # Các yếu tố cơ bản trên đường
    "vỉa hè đường phố việt nam",
    "lối sang đường dành cho người đi bộ",
    "đèn tín hiệu giao thông cho người đi bộ",
    "cột đèn đường",
    "biển báo giao thông",
    "dải phân cách đường",
    "lối đi dành cho người khuyết tật",
    
    # Các tình huống giao thông
    "người đi bộ qua đường",
    "người đi bộ chờ đèn đỏ",
    "người đi bộ tại ngã tư",
    "người đợi xe bus tại trạm",
    "người băng qua đường tại vạch kẻ",
    
    # Các chướng ngại vật
    "miệng cống thoát nước trên vỉa hè",
    "công trình đang thi công trên vỉa hè",
    "chướng ngại vật trên đường đi bộ",
    "xe máy đậu trên vỉa hè",
    
    # Các điểm định hướng
    "trạm xe buýt việt nam",
    "cột đèn giao thông tại ngã tư",
    "biển chỉ dẫn đường phố",
    "vạch kẻ đường cho người đi bộ",
    
    # Tình huống nguy hiểm
    "ổ gà trên đường",
    "đường ngập nước",
    "công trình đào đường",
    "xe máy chạy ngược chiều",
    
    # Hỗ trợ đặc biệt
    "gờ giảm tốc trên đường",
    "vạch dành cho người khiếm thị",
    "nút bấm tín hiệu cho người đi bộ",
    
    # Các từ khoá theo thời gian 
    "đường phố ban ngày",
    "giao thông ban đêm",
    "đường phố giờ cao điểm",
    "đường phố trời mưa",
    "giao thông trong sương mù",

    # Từ khóa bổ sung
    "giao thông thông minh",
    "hệ thống đèn giao thông tự động",
    "bãi đỗ xe thông minh",
    "cảm biến giao thông",
    "hệ thống giám sát giao thông",
    "đường phố xanh sạch đẹp",
    "giao thông công cộng hiện đại",
    "xe điện trên đường phố",
    "công nghệ giao thông tiên tiến",
    "hệ thống thoát nước đô thị",
    "an toàn giao thông cho trẻ em",
    "đường phố không rác thải",
    "giao thông bền vững",
    "hệ thống cảnh báo giao thông",
    "đèn đường năng lượng mặt trời"
]

all_results = []

def save_checkpoint(current_query, current_page, results):
    checkpoint = {
        'query': current_query,
        'page': current_page,
        'results': results
    }
    with open('checkpoint.json', 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False)

def load_checkpoint():
    try:
        with open('checkpoint.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def validate_image_info(image_info):
    required_fields = ['title', 'original_url', 'search_query']
    return all(image_info.get(field) for field in required_fields)

def cleanup_invalid_images():
    for file in os.listdir(IMAGES_DIR):
        file_path = os.path.join(IMAGES_DIR, file)
        try:
            if not imghdr.what(file_path):  # Kiểm tra file có phải ảnh không
                os.remove(file_path)
                logging.info(f"Removed invalid image: {file_path}")
        except Exception as e:
            logging.error(f"Error checking {file_path}: {str(e)}")

for query in tqdm(search_queries, desc="Đang xử lý từ khóa"):
    for page in range(0, 3):
        params = {
            "q": query,
            "engine": "google_images",
            "ijn": str(page),
            "api_key": "2514b459d190bed57a97218bcxxxxxxxxxxxxxxxxxxxxxxx",
            "location": "Vietnam",
            "safe": "off",
            "num": "200"
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            if "error" in results:
                print(f"Lỗi với từ khóa '{query}', trang {page}:", results["error"])
                continue

            if "images_results" in results:
                for image in tqdm(results["images_results"], desc=f"Đang tải ảnh cho '{query}' trang {page}"):
                    # Tải ảnh
                    saved_path = download_image(image.get('original'), query)
                    
                    # Lưu thông tin
                    image_info = {
                        'title': image.get('title'),
                        'original_url': image.get('original'),
                        'thumbnail_url': image.get('thumbnail'),
                        'source_website': image.get('source'),
                        'resolution': f"{image.get('original_width')}x{image.get('original_height')}",
                        'search_query': query,
                        'page_number': page,
                        'local_path': saved_path
                    }
                    all_results.append(image_info)

            time.sleep(2)

        except Exception as e:
            print(f"Lỗi xử lý từ khóa '{query}', trang {page}:", str(e))
            continue

# Lưu metadata dạng JSON (giữ nguyên Unicode)
with open('../output/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Lưu CSV với encoding phù hợp
df = pd.DataFrame(all_results)
df.to_csv('../output/traffic_images_dataset.csv', index=False, encoding='utf-8-sig')  # utf-8-sig để hỗ trợ Excel

# In thống kê
print(f"\nTổng số ảnh đã crawl: {len(all_results)}")
print(f"Số lượng ảnh theo từ khóa:")
print(df['search_query'].value_counts())