# OCR API with Flask and Ngrok

This repository provides an example of how to set up an OCR (Optical Character Recognition) API using Flask and expose it to the internet using Ngrok.

## Prerequisites

- Python 3.6+
- pip (Python package installer)
- Ngrok account (for creating a public URL)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```sh
    pip install -U transformers==4.44.2 bitsandbytes
    pip install -U huggingface_hub
    pip install flask flask-cors pyngrok flash_attn
    ```

## Setting Up the Flask API

1. Create a file named `app.py` with the following content:

    ```python
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/imc', methods=['POST'])
    def imc():
        data = request.json
        image_url = data.get('image_url')
        # Perform OCR on image_url
        result = perform_ocr_on_image(image_url)
        return jsonify({"response_message": result})

    def perform_ocr_on_image(image_url):
        # OCR logic here
        return "Sample OCR result"

    if __name__ == '__main__':
        app.run(port=5000)
    ```

2. Run the Flask API:

    ```sh
    python app.py
    ```

## Exposing the API with Ngrok

1. Install Ngrok and authenticate with your account:

    ```sh
    pip install pyngrok
    ngrok authtoken YOUR_NGROK_AUTH_TOKEN
    ```

2. Create a file named `ngrok_tunnel.py` with the following content:

    ```python
    from pyngrok import ngrok

    public_url = ngrok.connect(5000)
    print(f'Public URL: {public_url}')
    ```

3. Run the Ngrok tunnel:

    ```sh
    python ngrok_tunnel.py
    ```

    This will print a public URL (e.g., `https://abc123.ngrok.io`) that you can use to access your API.

## Using the OCR API

1. Create a file named `client.py` with the following content:

    ```python
    import requests

    def perform_ocr(image_path):
        response = requests.post(
            url="https://YOUR_NGROK_URL/imc",  # Replace with your Ngrok URL
            json={
                "image_url": image_path,
            }
        )

        print("Response time =", response.elapsed.total_seconds())

        if response.status_code == 200:
            return response.json().get("response_message")
        else:
            print("Error:", response.status_code, response.text)
            return None

    # Replace with your image path
    image_path = "https://uploads.nguoidothi.net.vn/content/a0a1cfac-64c2-47eb-a750-097a09008ccd.jpg"

    result = perform_ocr(image_path)

    if result:
        print("OCR Recognition Result:")
        print(result)
    ```

2. Run the client script:

    ```sh
    python client.py
    ```

    This will send a request to your API and print the OCR result.

## Conclusion

This repository demonstrates how to set up a simple OCR API using Flask and expose it to the internet using Ngrok. You can extend the OCR logic in `perform_ocr_on_image` function to integrate with any OCR library or service of your choice.