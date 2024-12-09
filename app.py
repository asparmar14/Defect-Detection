import io
import os
from PIL import Image
import torch
from flask import Flask, request, send_file, render_template

app = Flask(__name__)

# Automatically find the YOLO model in the directory
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    raise FileNotFoundError("Please place a YOLO model file in this directory!")

# Load the YOLO model
try:
    model_name = find_model()
    model = torch.hub.load(
        "WongKinYiu/yolov7",  # GitHub repo
        'custom',             # Custom model type
        model_name,           # Model name or path
        trust_repo=True,       # Trust the repository
        force_reload=True
    )
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Get predictions from the model
def get_prediction(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img], size=416)  # Includes Non-Maximum Suppression (NMS)

        # Save processed image to a BytesIO buffer
        buffer = io.BytesIO()
        results.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files.get('file')
        if not file:
            return "No file selected", 400

        try:
            img_bytes = file.read()
            buffer = get_prediction(img_bytes)

            # Send the processed image with a download option
            return send_file(
                buffer,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='detected_image.jpg'
            )
        except Exception as e:
            return f"Error processing the file: {e}", 500

    # Render the upload form
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
