"""
Factory Face Mask Detection — Flask backend.

Serves a webcam UI and exposes a /detect endpoint that runs a YOLOv8 model
on submitted frames and returns the detected boxes (mask / no_mask / improper_mask).

Built during an internship at Packages Limited.
"""

import os
import socket
import base64
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# ==============================
# Configuration
# ==============================
# Confidence threshold below which detections are ignored.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
# Inference image size (must match how the model was trained).
IMG_SIZE = int(os.environ.get("IMG_SIZE", 320))

# Friendly, UI-facing names for each raw model class.
LABEL_DISPLAY = {
    "mask": "Mask On",
    "no_mask": "No Mask",
    "improper_mask": "Improper Mask",
}

app = Flask(__name__)


# ==============================
# Load the YOLOv8 model
# ==============================
def load_model():
    """Find and load the first available trained model.

    Looks at MODEL_PATH first, then the trained checkpoint, then the
    exported ONNX file, so the app works both in development and after
    a fresh clone/deploy.
    """
    candidates = [
        os.environ.get("MODEL_PATH"),
        os.path.join("runs", "detect", "train2", "weights", "best.pt"),
        "best.onnx",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            print(f"Loading YOLOv8 model from: {path}")
            return YOLO(path)
    raise FileNotFoundError(
        "No model file found. Set MODEL_PATH or place 'best.onnx' in the project root."
    )


model = load_model()
class_labels = model.names
print("Class labels:", class_labels)


# ==============================
# HTTPS helpers (webcam requires a secure context)
# ==============================
def get_local_ip():
    """Return the machine's LAN IP, falling back to localhost."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except OSError:
        return "127.0.0.1"


def generate_self_signed_cert(cert_file, key_file):
    """Create a self-signed SSL certificate so the browser allows camera access."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Mask Detection App"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(get_local_ip()),
            ]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# ==============================
# Routes
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Run inference on a base64 image and return all detected boxes."""
    try:
        image_data = request.get_json()["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        np_arr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Could not decode image"})

        height, width = frame.shape[:2]
        results = model.predict(
            source=frame, verbose=False, conf=CONF_THRESHOLD, imgsz=IMG_SIZE
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = class_labels[int(box.cls[0])]
            detections.append({
                "label": label,
                "display": LABEL_DISPLAY.get(label, label),
                "confidence": float(box.conf[0]),
                "box": [x1, y1, x2, y2],
            })

        best = max(detections, key=lambda d: d["confidence"]) if detections else None
        return jsonify({
            "status": "success",
            "detections": detections,
            "image_width": width,
            "image_height": height,
            "prediction": best["label"] if best else "none",
            "confidence": best["confidence"] if best else 0.0,
        })

    except Exception as e:  # noqa: BLE001 - report any failure to the client
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


@app.route("/health")
def health_check():
    return jsonify({"status": "ok", "classes": list(class_labels.values())})


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("RENDER"):
        # Behind a platform proxy that terminates TLS.
        print(f" * Running on http://0.0.0.0:{port} (Render)")
        app.run(debug=False, host="0.0.0.0", port=port)
    else:
        if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
            print("Generating self-signed SSL certificate...")
            generate_self_signed_cert("cert.pem", "key.pem")
        print(f" * Running on https://127.0.0.1:{port} (Local Dev Server)")
        print(f" * On your network: https://{get_local_ip()}:{port}")
        app.run(debug=True, host="0.0.0.0", port=port, ssl_context=("cert.pem", "key.pem"))
