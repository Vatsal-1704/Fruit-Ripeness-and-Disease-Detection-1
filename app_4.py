from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLO model for fruit detection
fruit_detection_model = YOLO("/config/workspace/weights_3/best.pt")  # Change this path to the path of your YOLO model
banana_disease_detection_model = YOLO(
    "/config/workspace/train2/weights/best.pt")  # Path to YOLOv8 model for banana disease detection
mango_disease_detection_model = YOLO(
    "/config/workspace/train/weights/best.pt")  # Path to YOLOv8 model for mango disease detection
pomogranate_disease_detection_model = YOLO(
    "/config/workspace/train4/weights/best.pt")  # Path to YOLOv8 model for pomogranate disease detection


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Receive image data from the client
    image_data = request.json['image_data'].split(',')[1]  # Remove the data URL prefix

    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Perform object detection using YOLO
    results = fruit_detection_model(image)

    # Extract detection results
    detected_objects = []
    for result in results:
        boxes = result.boxes.xywh.cpu()  # xywh bbox list
        clss = result.boxes.cls.cpu().tolist()  # classes Id list
        names = result.names  # classes names list
        confs = result.boxes.conf.float().cpu().tolist()  # probabilities of classes

        for box, cls, conf in zip(boxes, clss, confs):
            detected_objects.append({'class': names[cls], 'bbox': box.tolist(), 'confidence': conf})

    return jsonify(detected_objects)


@app.route('/disease_detection')
def disease_detection():
    return render_template('disease_detection.html')


@app.route('/banana_detection', methods=['GET', 'POST'])
def banana_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(banana_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('banana_detection.html')


@app.route('/mango_detection', methods=['GET', 'POST'])
def mango_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(mango_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('mango_detection.html')

@app.route('/pomogranate_detection', methods=['GET', 'POST'])
def pomogranate_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(pomogranate_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('pomogranate_detection.html')


def detect_disease(model, image):
    result = model(image)
    class_names = []
    for result in result:
        probs = result.probs
        class_index = probs.top1
        class_name = result.names[class_index]
        score = float(probs.top1conf.cpu().numpy())
        class_names.append(class_name)
    return class_names


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # Read frame from camera
        if not success:
            break
        else:
            fruit_results = fruit_detection_model(frame)
            for result in fruit_results:
                im_array = result.plot()
                im = Image.fromarray(im_array[..., ::-1])
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
