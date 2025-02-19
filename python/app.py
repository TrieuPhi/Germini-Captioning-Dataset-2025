import pandas as pd
import requests
import time

def perform_ocr(image_url):
    response = requests.post(
        url="https://e555-34-125-45-105.ngrok-free.app/ocr",
        json={"image_url": image_url}
    )
    
    print(f"Processing {image_url}...")
    
    if response.status_code == 200:
        return response.json().get("response_message", "No caption generated")
    else:
        print("Error:", response.status_code, response.text)
        return "Error"

# Đọc file CSV
file_path = "./data/processed_traffic_images100.csv"
df = pd.read_csv(file_path)

# cột chứa URL hình ảnh tên là 'original_url'
df["caption"] = df["original_url"].apply(lambda url: perform_ocr(url) if pd.notna(url) else "No image")
    

