from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # so frontend can talk to backend

IMG_SIZE = 64  # same as training
# Assume MODEL is already loaded here

def preprocess_pil_image(pil_img, target_size):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((target_size, target_size))
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    pil_img = Image.open(file.stream)
    arr = preprocess_pil_image(pil_img, IMG_SIZE)

    preds = MODEL.predict(arr)
    top_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][top_idx])

    labels = ["A", "B", "C", "D", "E"]  # replace with your dataset labels
    predicted_sign = labels[top_idx]

    return jsonify({
        "sign": predicted_sign,
        "confidence": confidence
    })
