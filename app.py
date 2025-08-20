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
# Load Model
# ==============================
# ✅ Load your trained MobileNetV2 model once
model = load_model("mask-detector-model.h5")  # <-- replace with your actual model path
class_labels = ["mask", "no_mask"]  # update if you trained with more classes

# ==============================
# Helper Functions
# ==============================
def get_local_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return '127.0.0.1'

def generate_self_signed_cert(cert_file, key_file):
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
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My Company"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    cert = (x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256()))

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# Ensure cert.pem and key.pem exist
if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
    generate_self_signed_cert("cert.pem", "key.pem")

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        image_data = data['image']

        # decode base64 → image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # preprocess for MobileNetV2
        image = image.resize((224, 224))   # MobileNetV2 input size
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # run model prediction
        preds = model.predict(img_array)
        class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        prediction = class_labels[class_index]

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ==============================
# Run App
# ==============================
if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f" * Running on https://{local_ip}:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))

