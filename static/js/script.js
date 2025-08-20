// Global variables
let currentStream = null;
let detectionInterval = null;
let isCameraActive = false;

// DOM Elements
const videoElement = document.getElementById('video');
const statusElement = document.getElementById('status');
const predictionResult = document.getElementById('predictionResult');
const confidenceMeter = document.querySelector('.progress-bar');
const recommendationDiv = document.getElementById('recommendation');
const resultDiv = document.getElementById('result');

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    if (checkCameraSupport()) {
        await startCamera(); // start camera automatically
    }
});

// Check browser camera support
function checkCameraSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showUnsupportedMessage();
        return false;
    }
    return true;
}

function showUnsupportedMessage() {
    statusElement.textContent = "Camera API not supported in this browser. Please use Chrome, Firefox, or Edge.";
    statusElement.className = "alert alert-warning";
}

// Start camera function
async function startCamera() {
    if (!checkCameraSupport()) return;

    try {
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });

        videoElement.srcObject = currentStream;

        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });

        isCameraActive = true;
        statusElement.textContent = "Camera is ON. Detecting...";
        statusElement.className = "alert alert-success";

        if (!detectionInterval) {
            startDetectionLoop();
        }

    } catch (error) {
        handleCameraError(error);
    }
}

// Start detection loop
function startDetectionLoop() {
    detectionInterval = setInterval(async () => {
        if (!isCameraActive) return;

        try {
            const imageData = captureFrame();
            const detectionResult = await sendFrameForDetection(imageData);
            updateDetectionUI(detectionResult);

            // also show raw result in #result div
            if (detectionResult.prediction) {
                resultDiv.textContent = `${detectionResult.prediction} (confidence: ${(detectionResult.confidence * 100).toFixed(1)}%)`;
            }

        } catch (error) {
            console.error("Frame processing error:", error);
            statusElement.textContent = "Error processing frame";
            statusElement.className = "alert alert-danger";
        }
    }, 500); // every 500ms
}

// Capture frame from video
function captureFrame() {
    if (!videoElement.videoWidth || !videoElement.videoHeight) {
        throw new Error("Video not ready yet");
    }
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Send frame to server for detection
async function sendFrameForDetection(imageData) {
    // imageData is already base64 data URL from captureFrame()
    let base64Image = imageData.split(",")[1]; // strip "data:image/jpeg;base64,"

    const response = await fetch('/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: base64Image })
    });

    return await response.json();
}

// Update UI with detection results
function updateDetectionUI(result) {
    if (result.status !== 'success') {
        statusElement.textContent = `Detection error: ${result.message || 'Unknown error'}`;
        statusElement.className = "alert alert-warning";
        return;
    }

    const prediction = result.prediction;
    const confidence = Math.round(result.confidence * 100);

    predictionResult.textContent = formatPredictionText(prediction);
    predictionResult.className = getPredictionClass(prediction);

    confidenceMeter.style.width = `${confidence}%`;
    confidenceMeter.textContent = `${confidence}%`;

    updateRecommendation(prediction);
}

function formatPredictionText(prediction) {
    return prediction.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

function getPredictionClass(prediction) {
    const classes = {
        'correct_mask': 'text-success',
        'incorrect_mask': 'text-warning',
        'no_mask': 'text-danger'
    };
    return classes[prediction] || '';
}

function updateRecommendation(prediction) {
    const messages = {
        'correct_mask': 'Great job! You\'re wearing your mask correctly.',
        'incorrect_mask': 'Please adjust your mask to cover both nose and mouth.',
        'no_mask': 'No mask detected. Please wear a mask for safety.'
    };
    recommendationDiv.textContent = messages[prediction] || '';
    recommendationDiv.className = `alert ${getAlertClass(prediction)}`;
    recommendationDiv.style.display = 'block';
}

function getAlertClass(prediction) {
    const classes = {
        'correct_mask': 'alert-success',
        'incorrect_mask': 'alert-warning',
        'no_mask': 'alert-danger'
    };
    return classes[prediction] || 'alert-info';
}

function handleCameraError(error) {
    console.error("Camera Error:", error);
    const errorMessages = {
        'NotAllowedError': 'Please allow camera permissions',
        'NotFoundError': 'No camera device found',
        'NotReadableError': 'Camera is already in use',
        'OverconstrainedError': 'Camera doesn\'t support requested settings',
        'SecurityError': 'Camera access blocked for security reasons',
        'AbortError': 'Camera access was aborted'
    };
    statusElement.textContent = errorMessages[error.name] || 'Error accessing camera';
    statusElement.className = "alert alert-danger";
}

// Stop camera when leaving page
window.addEventListener('beforeunload', () => {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
});
