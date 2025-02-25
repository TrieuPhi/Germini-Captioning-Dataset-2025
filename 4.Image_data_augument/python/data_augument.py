import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from io import BytesIO
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Định nghĩa các đường dẫn
INPUT_CSV = "./csv_with_captions/valid_urls_dataset_v12.csv"
OUTPUT_DIR = "./augmented"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "captions_augmented.csv")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Định nghĩa các augmentation transforms
transforms = A.Compose([
    # Biến đổi về cường độ pixel
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
    ], p=0.5),
    
    # Biến đổi không gian
    A.OneOf([
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            p=0.5
        ),
    ], p=0.5),
    
    # Thêm nhiễu hoặc làm mờ nhẹ
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.3),
    
    # Random crop với tỷ lệ cao
    A.RandomResizedCrop(
        height=512,
        width=512,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=0.5
    ),
    
    # Đảm bảo kích thước output đồng nhất
    A.Resize(512, 512, p=1.0)
])

def download_image(url, timeout=10):
    """Tải ảnh từ URL"""
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        if response.status_code != 200:
            raise Exception(f"HTTP error {response.status_code}")
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logging.error(f"Lỗi khi tải ảnh từ {url}: {str(e)}")
        return None

def process_image(image):
    """Xử lý ảnh để chuẩn bị cho augmentation"""
    if image.mode not in ['RGB', 'L']:
        image = image.convert('RGB')
    
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    return image

def save_image(image, path):
    """Lưu ảnh với xử lý lỗi"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(path)
        return True
    except Exception as e:
        logging.error(f"Lỗi khi lưu ảnh {path}: {str(e)}")
        return False

def process_single_row(row, idx, output_dir):
    """Xử lý một hàng dữ liệu"""
    results = []
    image_url = row['original_url']
    if pd.isna(image_url):
        return results

    original_filename = f"image_{idx}.jpg"
    new_original_path = os.path.join(output_dir, "images", original_filename)

    # Tải và lưu ảnh gốc
    original_image = download_image(image_url)
    if original_image is None:
        return results

    if save_image(original_image, new_original_path):
        results.append({
            'original_url': row['original_url'],
            'source_website': row['source_website'],
            'resolution': row['resolution'],
            'search_query': row['search_query'],
            'local_path': new_original_path,
            'short_caption': row['short_caption']
        })

    # Tạo augmented images
    processed_image = process_image(original_image)
    for aug_idx in range(3):  # num_augmentations = 3
        augmented = transforms(image=processed_image)['image']
        name, ext = os.path.splitext(original_filename)
        new_name = f"{name}_aug_{aug_idx}{ext}"
        new_path = os.path.join(output_dir, "images", new_name)
        
        if save_image(augmented, new_path):
            results.append({
                'original_url': row['original_url'],
                'source_website': row['source_website'],
                'resolution': row['resolution'],
                'search_query': row['search_query'],
                'local_path': new_path,
                'short_caption': row['short_caption']
            })

    return results

def main():
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO)
    
    # Đọc file CSV gốc
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Đọc được {len(df)} ảnh từ file CSV")
    
    all_results = []
    
    # Xử lý đa luồng
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(process_single_row, row, idx, OUTPUT_DIR): idx
            for idx, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="Processing images"):
            results = future.result()
            all_results.extend(results)
    
    # Tạo DataFrame mới và lưu
    new_df = pd.DataFrame(all_results)
    new_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    logging.info(f"Đã lưu file CSV mới tại: {OUTPUT_CSV}")
    logging.info(f"Tổng số ảnh (gốc + augmented): {len(new_df)}")

if __name__ == "__main__":
    main()