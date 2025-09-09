import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import os
import io
import json
import base64
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import requests

# ---------- CONFIG ----------
MODEL_DIR = "model"
H5_MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

IMG_SIZE = 64
CHANNEL_LAST = True  # most TF models expect channels_last

# ---------- APP ----------
app = Flask(__name__)
CORS(app)  # enable CORS for all routes

# ---------- HELPERS ----------
def load_labels(path=LABELS_PATH):
    if not os.path.exists(path):
        print(f"labels.json not found at: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
        print(f"Loaded labels: {labels}")
        return labels

def preprocess_pil_image(pil_img: Image.Image, img_size=IMG_SIZE):
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if not CHANNEL_LAST:
        arr = np.transpose(arr, (2, 0, 1))
    return arr

def decode_base64_image(img_base64):
    try:
        decoded = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(decoded))
        return img
    except Exception:
        return None

# ---------- MODEL LOADING ----------
class TFModelWrapper:
    def __init__(self):
        self.model_type = None
        self.keras_model = None
        self.tflite_interpreter = None
        self.labels = load_labels()
        self.input_shape = None
        self.load_model()

    def load_model(self):
        if os.path.exists(H5_MODEL_PATH):
            print("Loading Keras model:", H5_MODEL_PATH)
            self.keras_model = tf.keras.models.load_model(H5_MODEL_PATH)
            self.model_type = "h5"
            try:
                self.input_shape = self.keras_model.input_shape
            except Exception:
                self.input_shape = None
        elif os.path.exists(TFLITE_MODEL_PATH):
            print("Loading TFLite model:", TFLITE_MODEL_PATH)
            self.tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            self.tflite_interpreter.allocate_tensors()
            self.model_type = "tflite"
        else:
            raise FileNotFoundError("No model found. Place .h5 or .tflite in 'model/' folder.")

    def predict(self, image_np: np.ndarray):
        if self.model_type == "h5":
            x = np.expand_dims(image_np, axis=0)
            preds = self.keras_model.predict(x)[0]
        elif self.model_type == "tflite":
            inp_details = self.tflite_interpreter.get_input_details()
            out_details = self.tflite_interpreter.get_output_details()
            inp = np.expand_dims(image_np, axis=0).astype(np.float32)
            if tuple(inp.shape) != tuple(inp_details[0]['shape']):
                try:
                    self.tflite_interpreter.resize_tensor_input(inp_details[0]['index'], inp.shape)
                    self.tflite_interpreter.allocate_tensors()
                except Exception:
                    pass
            self.tflite_interpreter.set_tensor(inp_details[0]['index'], inp)
            self.tflite_interpreter.invoke()
            preds = self.tflite_interpreter.get_tensor(out_details[0]['index'])[0]
        else:
            raise RuntimeError("Model not loaded")

        preds = np.asarray(preds, dtype=np.float32)
        if preds.ndim == 0:
            preds = np.array([float(preds)])

        if not np.all((preds >= 0) & (preds <= 1)) or not np.isclose(np.sum(preds), 1.0):
            e = np.exp(preds - np.max(preds))
            preds = e / e.sum()

        top_k = 5
        top_idx = preds.argsort()[::-1][:top_k]
        top = [(self.labels.get(str(idx), str(idx)) if self.labels else str(idx), float(preds[idx]))
               for idx in top_idx]

        return {"predictions": preds.tolist(), "top": top}

# instantiate model once
MODEL = TFModelWrapper()
print(f"Model type: {MODEL.model_type}")
print(f"Labels loaded: {MODEL.labels is not None}")

# ---------- ROUTES ----------

@app.route("/", methods=["GET"])
def index():
    # Serve frontend HTML page
    return render_template("frontend.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_type": MODEL.model_type})

@app.route("/labels", methods=["GET"])
def labels():
    if MODEL.labels:
        return jsonify({"labels": MODEL.labels})
    return jsonify({"error": "labels.json not found"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    img = None

    # Check multipart/form-data for image
    if "image" in request.files:
        file = request.files["image"]
        try:
            img = Image.open(file.stream)
        except Exception as e:
            return jsonify({"error": "Cannot open image", "detail": str(e)}), 400

    # Support base64 images in JSON
    elif request.is_json:
        data = request.get_json()
        img_base64 = data.get("image_base64")
        if not img_base64:
            return jsonify({"error": "No 'image_base64' in JSON"}), 400
        img = decode_base64_image(img_base64)
        if img is None:
            return jsonify({"error": "Invalid base64 image"}), 400

    else:
        return jsonify({"error": "No image provided"}), 400

    # Preprocess and predict
    img_arr = preprocess_pil_image(img, IMG_SIZE)
    result = MODEL.predict(img_arr)

    if MODEL.labels is None:
        return jsonify({"error": "Labels not loaded"}), 500

    top_predictions = result.get("top", [])
    if not top_predictions:
        return jsonify({"error": "Prediction failed"}), 500

    raw_label = top_predictions[0][0]  # Could be label or index as string
    confidence = top_predictions[0][1]

    # Convert numeric label to actual letter using labels.json mapping
    if raw_label.isdigit():
        label = MODEL.labels.get(raw_label, raw_label)
    else:
        label = raw_label

    return jsonify({
        "sign": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(debug=True)