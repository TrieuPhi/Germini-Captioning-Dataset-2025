import os
import logging
import warnings
import urllib3
import pandas as pd
import time
from tqdm import tqdm  # Để hiển thị progress bar

# Suppress all warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger().setLevel(logging.ERROR)

import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO

def init_gemini(api_key):
    """Initialize Gemini API"""
    genai.configure(api_key=api_key)

def load_image_from_url(url):
    """Load image from URL with resize"""
    try:
        response = requests.get(url, timeout=10, verify=False)  # Bỏ qua SSL verify
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # Resize image if too large
        max_size = (800, 800)  # Giới hạn kích thước tối đa
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def get_prediction(image_url, prompt, max_retries=3):
    """Get prediction with retries"""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            image = load_image_from_url(image_url)
            if image is None:
                return None

            response = model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error generating content: {e}")
                return None
            time.sleep(2 * (attempt + 1))  # Exponential backoff

def process_dataset(csv_path, prompt, batch_size=10):
    """Process dataset with longer delays"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        
        backup_path = csv_path.replace('.csv', '_backup.csv')
        df.to_csv(backup_path, index=False)
        print(f"Created backup at {backup_path}")
        
        for idx in tqdm(range(len(df))):
            if pd.notna(df.at[idx, 'short_caption']):
                continue
                
            url = df.at[idx, 'original_url']
            caption = get_prediction(url, prompt)
            
            if caption:
                df.at[idx, 'short_caption'] = caption
                
            if idx % batch_size == 0:
                df.to_csv(csv_path, index=False)
                print(f"\nSaved progress at row {idx}")
                time.sleep(2)  # Tăng delay giữa các batch
            
        df.to_csv(csv_path, index=False)
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

# Optimized prompt
OPTIMIZED_PROMPT = """
You are a visually impaired person listening to a description of the surrounding traffic situation. Briefly and objectively describe the following criteria in a short paragraph:  
        **Traffic condition**:  
            - Describe the current traffic condition in one simple sentence, focusing on the main vehicle, traffic signs, traffic lights, people, and the scene in the image.  
        **Position of fixed objects**:  
            - Clearly state positions such as "left", "right", "ahead", "center", "on the side" for fixed objects like traffic signs, traffic lights, or police booths.  
            - If there are traffic signs or lights, mention their content clearly.  
        **Lane and movement direction**:  
            - Specify whether vehicles are moving in the same or opposite direction from me (based on the image's perspective).  
            - If a vehicle is crossing, state its direction (e.g., "from left to right", "from right to left").  
        **Viewpoint in the image**:  
            - Determine your position relative to the camera view (e.g., standing on the sidewalk, in the middle of the road, or viewing from a distance). Refer to yourself as "you".  
        **Safe mobility**:  
            - Accurately identify the "left", "right", "ahead", or "center" position of lanes with sidewalks, pedestrian crossings, or lanes without obstacles that affect safe movement.  
        **Include the following constraints**:  
            - Use only simple sentences with clear subjects, verbs, and objects.  
            - Use natural, spoken-like language that is easy to understand.  
            - Do not describe obvious information.  
            - Separate ideas with periods (.) and do not use commas (,) or semicolons (;) to connect sentences.  
            - Do not add emotion or speculation. Do not exceed 20 words. If longer, shorten the sentence.  
            - All sentences must be in a single line without line breaks.
"""

if __name__ == "__main__":
    # Initialize Gemini
    API_KEY = "AIzaSxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key
    init_gemini(API_KEY)
    
    # Process dataset
    csv_path = "./cleaned_dataset.csv"
    process_dataset(csv_path, OPTIMIZED_PROMPT)