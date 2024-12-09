import io
import os
from PIL import Image
import torch
from flask import Flask, request, send_file, render_template

app = Flask(__name__)
'''

# Automatically find the YOLO model in the directory
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    raise FileNotFoundError("Please place a YOLO model file in this directory!")
    '''

# Load the YOLO model
try:
    #model_name = find_model()
    model_name = "last_best.pt"
    model = torch.hub.load(
        "WongKinYiu/yolov7",  # GitHub repo
        'custom',             # Custom model type
        model_name,           # Model name or path
        trust_repo=True,       # Trust the repository
        force_reload=True
    )
    model.eval()
    # Set model to use FP16 if supported for better memory efficiency
    if torch.cuda.is_available():
        model.half()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Get predictions from the model
def get_prediction(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))

        # Resize the image before passing it to the model to save memory
        img = img.resize((416, 416))  # Resize to YOLOv7 input size

        # Convert the image to a tensor
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()  # Convert to tensor (C, H, W)
        img_tensor /= 255.0  # Normalize image to [0, 1]

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Send the image tensor to the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        model.to(device)

        with torch.no_grad():  # Disable gradient calculation for inference
            results = model(img_tensor)  # Inference

        # Save processed image to a BytesIO buffer
        buffer = io.BytesIO()
        results.render()  # Render the results to the image
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
    app.run(host="0.0.0.0", port=5000, debug=True)
