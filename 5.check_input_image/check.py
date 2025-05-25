import os
import warnings
import urllib3
import time
from datetime import datetime
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API keys
API_KEYS = [
    "AIzaSyBnvv2aFHTMY...............",
    # Add other keys as needed or keep the existing list
]

MODEL_NAME = "gemini-2.0-flash-lite"  # Default model name

class APIManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.request_counts = {key: 0 for key in api_keys}
        self.minute_start_times = {key: datetime.now() for key in api_keys}
        self.rpm_limit = 15  # Requests per minute

    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]

    def switch_api_key(self):
        old_key = self.get_current_api_key()
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.get_current_api_key()
        print(f"Switching from API key {old_key[-5:]} to {new_key[-5:]}")
        init_gemini(new_key)
        return new_key

    def track_request(self):
        key = self.get_current_api_key()
        now = datetime.now()
        
        # Reset minute counter if needed
        if (now - self.minute_start_times[key]).total_seconds() > 60:
            self.request_counts[key] = 0
            self.minute_start_times[key] = now
            
        # Increment counter
        self.request_counts[key] += 1
        
        # Check if we need to switch keys due to rate limits
        if self.request_counts[key] >= self.rpm_limit:
            return self.switch_api_key()
        
        return key

def init_gemini(api_key):
    """Initialize Gemini API"""
    genai.configure(api_key=api_key)

def load_image(image_path):
    """Load image from local path or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # It's a URL
            response = requests.get(image_path, timeout=10, verify=False)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            # It's a local file path
            image = Image.open(image_path)
        
        # Resize image if too large
        max_size = (800, 800)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def is_traffic_related(image_path, api_manager=None, max_retries=3, verbose=True):
    """
    Check if an image is related to traffic/transportation.
    
    Args:
        image_path: Local file path or URL of the image to check
        api_manager: Instance of APIManager for handling API keys (optional)
        max_retries: Maximum number of retry attempts
        verbose: Whether to print progress messages
        
    Returns:
        dict: {'is_traffic_related': bool, 'message': str, 'time_taken': float}
    """
    # Initialize API manager if not provided
    if api_manager is None:
        api_manager = APIManager(API_KEYS)
        init_gemini(api_manager.get_current_api_key())
    
    # Traffic relevance check prompt
    check_prompt = """Analyze this image and determine if it's related to traffic, transportation, 
                   roads, vehicles, or any traffic situation. 
                   Respond with ONLY 'YES' if it is traffic-related, or 'NO' if it is not."""
    
    # Start timing the entire process
    overall_start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            # Track request and get current API key
            current_key = api_manager.track_request()
            
            # Load the image
            image_start_time = time.time()
            image = load_image(image_path)
            image_time = time.time() - image_start_time
            
            if image is None:
                overall_time = time.time() - overall_start_time
                return {
                    'is_traffic_related': False,
                    'message': "Không thể tải hình ảnh, vui lòng kiểm tra đường dẫn.",
                    'time_taken': overall_time,
                    'load_time': image_time
                }

            if verbose:
                print(f"Checking if image is traffic-related: {image_path}")
                print(f"Image loaded in {image_time:.2f} seconds")
            
            # Start timing the API call
            api_start_time = time.time()
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content([check_prompt, image])
            result = response.text.strip().lower()
            api_time = time.time() - api_start_time
            
            if verbose:
                print(f"API response received in {api_time:.2f} seconds")
            
            is_traffic_related = 'yes' in result
            overall_time = time.time() - overall_start_time
            
            if is_traffic_related:
                if verbose:
                    print(f"✓ Image is traffic-related (total time: {overall_time:.2f}s)")
                return {
                    'is_traffic_related': True,
                    'message': None,
                    'time_taken': overall_time,
                    'api_time': api_time,
                    'load_time': image_time
                }
            else:
                if verbose:
                    print(f"✗ Image is NOT traffic-related (total time: {overall_time:.2f}s)")
                return {
                    'is_traffic_related': False,
                    'message': "Tôi là model được sử dụng cho dữ liệu giao thông, vui lòng hỏi bức hình liên quan đến giao thông",
                    'time_taken': overall_time,
                    'api_time': api_time,
                    'load_time': image_time
                }
                
        except Exception as e:
            error_str = str(e)
            if verbose:
                print(f"Attempt {attempt+1}/{max_retries}: {error_str}")
            
            # Switch API key if rate limit error
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                api_manager.switch_api_key()
            
            if attempt == max_retries - 1:
                overall_time = time.time() - overall_start_time
                return {
                    'is_traffic_related': False,
                    'message': "Lỗi khi phân tích hình ảnh, vui lòng thử lại sau.",
                    'time_taken': overall_time,
                    'error': error_str
                }
                
            time.sleep(2 * (attempt + 1))
    
    overall_time = time.time() - overall_start_time
    return {
        'is_traffic_related': False,
        'message': "Quá thời gian phân tích hình ảnh, vui lòng thử lại.",
        'time_taken': overall_time
    }

# Example usage
if __name__ == "__main__":
    api_manager = APIManager(API_KEYS)
    init_gemini(api_manager.get_current_api_key())
    
    start_time = time.time()
    print(f"Starting analysis at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Example with a local file path
    image_path = "./img/caycoi.jpg"
    result = is_traffic_related(image_path, api_manager)
    
    total_time = time.time() - start_time
    print(f"\nResults:")
    print(f"- Using model: {MODEL_NAME}")
    print(f"- Analysis completed in {total_time:.2f} seconds")
    print(f"- API processing time: {result.get('api_time', 'N/A'):.2f}s")
    print(f"- Image loading time: {result.get('load_time', 'N/A'):.2f}s")
    
    if result['is_traffic_related']:
        print("- Status: Image is traffic-related! Processing can continue.")
    else:
        print(f"- Status: Not traffic-related: {result['message']}")