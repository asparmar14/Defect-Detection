import io
import os
import gdown
from PIL import Image
import torch
from flask import Flask, request, send_file, render_template

app = Flask(__name__)

# Function to download the model from Google Drive
def download_model():
    # Replace with your actual file ID from the Google Drive model link
    file_id = '1E3EnO0bmCf4ill68GqImwjDDB6mavQ3v'  # Your model file ID
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # Download the model file
    output = 'last_best.pt'  # The file will be saved as last_best.pt
    gdown.download(url, output, quiet=False)

# Download the model if it doesn't already exist
if not os.path.exists('last_best.pt'):
    download_model()

# Load the YOLO model
model = 'last_best.pt'
model.eval()  # Set the model to evaluation mode

# Get predictions from the model
def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))  # Open image from bytes
    results = model([img], size=640)  # Perform inference with the model (includes NMS)
    
    # Save processed image to a BytesIO buffer
    buffer = io.BytesIO()
    results.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files.get('file')
        if not file:
            return "No file selected", 400

        img_bytes = file.read()  # Read the image bytes
        buffer = get_prediction(img_bytes)  # Get the processed image with predictions

        # Send the processed image with a download option
        return send_file(
            buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='detected_image.jpg'
        )

    # Render the upload form (GET request)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
