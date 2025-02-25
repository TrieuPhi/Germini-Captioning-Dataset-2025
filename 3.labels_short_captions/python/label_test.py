import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
import os
import logging

# Suppress warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger().setLevel(logging.ERROR)

def init_gemini(api_key):
    """Initialize Gemini API"""
    genai.configure(api_key=api_key)

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def get_prediction(image_url, prompt):
    """Get prediction from Gemini Vision model"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Load image from URL
    image = load_image_from_url(image_url)
    if image is None:
        return None

    try:
        # Generate content
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

# Optimized prompt
OPTIMIZED_PROMPT = """Describe the image naturally, as if a person were observing and explaining it. Keep it concise (2-3 sentences) but informative.  

Some aspects to consider if present:  
- Traffic: What vehicles are in the scene? Is traffic dense or light?  
- Roads: Are they clean, well-maintained, or obstructed? Any road signs or markings?  
- Traffic signals: Is the light red, green, or yellow? Are people stopping or moving?  
- People: Are there pedestrians? Are they wearing helmets or carrying umbrellas?  
- Weather: Is it sunny, rainy, foggy, or nighttime?  

Respond in Vietnamese naturally, as if describing the scene to someone else.
"""


if __name__ == "__main__":
    # Initialize Gemini
    API_KEY = "AIxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key
    init_gemini(API_KEY)
    
    # Test with a single image URL
    image_url = "https://suckhoedoisong.qltns.mediacdn.vn/324455921873985536/2025/1/10/dendo04-1736499844800-173649984491467143496.jpg"
    
    # Get prediction
    caption = get_prediction(image_url, OPTIMIZED_PROMPT)
    
    if caption:
        print("Generated caption:")
        print(caption)
    else:
        print("Failed to generate caption")