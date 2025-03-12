from flask import Flask, request, jsonify
import base64
import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

def initialize_client():
    """Initialize the Inference HTTP Client with API credentials."""
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=os.getenv("ROBOFLOW_API_KEY")  # Ensure this is valid
    )

def run_acne_detection(client, image_base64):
    """Send the base64-encoded image to the inference API and return the processed result."""
    result = client.run_workflow(
        workspace_name=os.getenv("ROBOFLOW_WORKSPACE_NAME"),
        workflow_id=os.getenv("ROBOFLOW_WORKFLOW_ID"),
        images={"image": image_base64},  # Pass base64-encoded string
        use_cache=True
    )
    return result[0]['bounding_box_visualization_1']

def process_acne_image(image_base64):
    """Process the base64 image and return the processed image as base64."""
    client = initialize_client()
    processed_base64 = run_acne_detection(client, image_base64)  # Pass base64 string
    return processed_base64

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    image_base64 = data.get('image_base64')
    
    if not image_base64:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Process the image
        processed_base64 = process_acne_image(image_base64)
        
        return jsonify({'processed_image_base64': processed_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
