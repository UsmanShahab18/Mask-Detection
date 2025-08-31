from flask import Flask, request, jsonify, render_template
import socket
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# Load Model and Class Labels
# ==============================
model = None
class_labels = None
IMG_SIZE = 150 # Must match the size used in train_model.py

try:
    # Load the trained model
    model = load_model("mymodel.h5")
    print("Model loaded successfully.")
    
    # Get the class labels from the model's metadata (if available) or assume alphabetical order
    if hasattr(model, 'class_indices'):
        class_indices = model.class_indices
        class_labels = sorted(class_indices, key=class_indices.get)
        print("Class labels loaded from model metadata.")
    else:
        # Fallback to the known alphabetical order from the training script
        class_labels = ["mask", "no_mask"]
        print("Using default class labels: ['mask', 'no_mask']")

except Exception as e:
    print(f"Error loading model or class labels: {e}")
    # We exit here because the app cannot function without a model
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
    """
    Generates a self-signed SSL certificate and private key.
    This is required for the webcam to work on a secure connection (https).
    """
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
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(get_local_ip())]), critical=False)
        .sign(key, hashes.SHA256()))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# Ensure cert.pem and key.pem exist for HTTPS
# The check below is for local development only and will not run on Render
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
    """
    Receives an image via POST request, runs the model prediction, and returns the result.
    """
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Check if the string contains a comma before splitting.
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
            
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        
        if not isinstance(preds, np.ndarray) or preds.size == 0:
            return jsonify({"status": "error", "message": "Model prediction returned invalid output."})

        # For binary classification with sigmoid, the output is a single value
        if preds[0][0] >= 0.5:
            prediction = "no_mask"
            confidence = float(preds[0][0])
        else:
            prediction = "mask"
            confidence = 1.0 - float(preds[0][0])
            
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/health')
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "ok", "message": "App is running"})

# ==============================
# Run App
# ==============================
if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f" * Running on https://{local_ip}:5000 (Local Dev Server)")
    # This is for local development only. Render uses Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
