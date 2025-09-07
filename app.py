from flask import Flask, request, jsonify, render_template
import socket
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
from ultralytics import YOLO

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# Load YOLOv8 Model
# ==============================
model = None
try:
    # FIXED PATH (your model is inside runs/detect/train2/weights/)
    model_path = os.path.join("runs", "detect", "train2", "weights", "best.pt")
    model = YOLO(model_path)
    print(f"YOLOv8 model loaded from {model_path}")

    # Class labels from training
    class_labels = model.names
    print("Class labels:", class_labels)

except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()


# ==============================
# Helper Functions
# ==============================
def get_local_ip():
    """Returns the local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return '127.0.0.1'


def generate_self_signed_cert(cert_file, key_file):
    """Generates a self-signed SSL certificate and private key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
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
            x509.IPAddress(get_local_ip())
        ]), critical=False)
        .sign(key, hashes.SHA256()))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# Ensure cert.pem and key.pem exist for HTTPS
if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
    print("Generating self-signed SSL certificate...")
    generate_self_signed_cert("cert.pem", "key.pem")


# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Receives an image via POST request, runs YOLOv8 prediction, and returns the result."""
    try:
        data = request.get_json()
        image_data = data['image']

        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)

        # Convert image bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLOv8 inference
        results = model.predict(source=frame, verbose=False, conf=0.25, imgsz=320)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = class_labels[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": confidence
            })

        if detections:
            best_detection = max(detections, key=lambda d: d['confidence'])
            return jsonify({
                "status": "success",
                "prediction": best_detection['label'],
                "confidence": best_detection['confidence']
            })
        else:
            return jsonify({
                "status": "success",
                "prediction": "none",
                "confidence": 0.0
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "message": "App is running"})


# ==============================
# Run App
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("RENDER"):
        print(f" * Running on http://0.0.0.0:{port} (Render)")
        app.run(debug=False, host="0.0.0.0", port=port)
    else:
        print(f" * Running on https://127.0.0.1:{port} (Local Dev Server)")
        app.run(debug=True, host="0.0.0.0", port=port, ssl_context=('cert.pem', 'key.pem'))
