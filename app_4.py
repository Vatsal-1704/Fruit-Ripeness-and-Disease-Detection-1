from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLO model for fruit detection
fruit_detection_model = YOLO(r"C:\Users\vatsal\Desktop\project\NEW MODEL PROJECT\Fruit-Ripeness-and-Disease-Detection\weights_3\best.pt")  # Change this path to the path of your YOLO model
banana_disease_detection_model = YOLO(
    r"C:\Users\vatsal\Desktop\project\NEW MODEL PROJECT\Fruit-Ripeness-and-Disease-Detection\train2\weights\best.pt")  # Path to YOLOv8 model for banana disease detection
mango_disease_detection_model = YOLO(
    r"C:\Users\vatsal\Desktop\project\NEW MODEL PROJECT\Fruit-Ripeness-and-Disease-Detection\train\weights\best.pt")  # Path to YOLOv8 model for mango disease detection
pomogranate_disease_detection_model = YOLO(
    r"C:\Users\vatsal\Desktop\project\NEW MODEL PROJECT\Fruit-Ripeness-and-Disease-Detection\train4\weights\best.pt")  # Path to YOLOv8 model for pomogranate disease detection




def check_templates():
    """Ensure all required templates exist"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    required_templates = [
        'index.html', 
        'fruit_detection.html', 
        'disease_detection.html',
        'banana_detection.html',
        'mango_detection.html',
        'pomogranate_detection.html',
        'uploaded_image.html',
        'nutrition.html'
    ]
    
    missing_templates = []
    for template in required_templates:
        if not os.path.exists(os.path.join(templates_dir, template)):
            missing_templates.append(template)
    
    if missing_templates:
        print(f"Warning: The following templates are missing: {', '.join(missing_templates)}")
        return False
    return True





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
            if "bb" not in file.filename:
                return render_template('banana_detection.html', alert="This fruit is not a banana.")
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            # Detect diseases with enhanced function
            disease_results = detect_disease(banana_disease_detection_model, img, 'banana')
            # Estimate ripeness
            ripeness = estimate_ripeness(img, 'banana')
            # Get ripeness category for nutrition information
            ripeness_category = get_ripeness_category(ripeness)
            # Convert image to base64 for display
            img_str = image_to_base64(img)

            return render_template('uploaded_image.html', 
                                  img_str=img_str, 
                                  disease_results=disease_results, 
                                  fruit='banana',
                                  ripeness=ripeness,
                                  ripeness_category=ripeness_category,
                                  nutrition=NUTRITION_INFO['banana'][ripeness_category],
                                  disease_info=DISEASE_INFO,
                                  storage_recommendations=STORAGE_RECOMMENDATIONS['banana'][ripeness_category])
    return render_template('banana_detection.html')


@app.route('/mango_detection', methods=['GET', 'POST'])
def mango_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            if "mg" not in file.filename:
                return render_template('mango_detection.html', alert="This fruit is not a mango.")
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            disease_results = detect_disease(mango_disease_detection_model, img, 'mango')
            ripeness=estimate_ripeness(img, 'mango')
            ripeness_category = get_ripeness_category(ripeness)
            img_str = image_to_base64(img)
            
            
            return render_template('uploaded_image.html', 
                                  img_str=img_str,
                                  disease_results=disease_results,
                                  fruit='mango',
                                  ripeness=ripeness,
                                  ripeness_category=ripeness_category,
                                  nutrition=NUTRITION_INFO['mango'][ripeness_category],
                                  disease_info=DISEASE_INFO,
                                  storage_recommendations=STORAGE_RECOMMENDATIONS['mango'][ripeness_category])
    return render_template('mango_detection.html')


@app.route('/pomogranate_detection', methods=['GET', 'POST'])
def pomogranate_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            if "pg" not in file.filename:
                return render_template('pomogranate_detection.html', alert="This fruit is not a pomogranate.")
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            disease_results = detect_disease(pomogranate_disease_detection_model, img, 'pomogranate')
            ripeness=estimate_ripeness(img, 'pomogranate')
            ripeness_category = get_ripeness_category(ripeness)
            img_str = image_to_base64(img)
            
            
            return render_template('uploaded_image.html', 
                                  img_str=img_str, 
                                  disease_results=disease_results,
                                  fruit='pomogranate',
                                  ripeness=ripeness,
                                  ripeness_category=ripeness_category,
                                  nutrition=NUTRITION_INFO['pomogranate'][ripeness_category],
                                  disease_info=DISEASE_INFO,
                                  storage_recommendations=STORAGE_RECOMMENDATIONS['pomogranate'][ripeness_category])
    return render_template('pomogranate_detection.html')


@app.route('/nutrition/<fruit>')
def nutrition_info(fruit):
    if fruit not in NUTRITION_INFO:
        return render_template('error.html', message=f"Nutrition information for {fruit} is not available")
    
    return render_template('nutrition.html', 
                          fruit=fruit,
                          nutrition=NUTRITION_INFO[fruit])



def detect_disease(model, image, fruit_type):
    """
    Detect diseases in fruits using trained YOLO models
    
    Parameters:
    model (YOLO): YOLOv8 model for disease detection
    image (PIL.Image): Image of the fruit
    fruit_type (str): Type of fruit ('banana', 'mango', or 'pomogranate')
    
    Returns:
    list: List of detected diseases with confidence scores
    """
    result = model(image)
    disease_results = []
    
    for r in result:
        if hasattr(r, 'probs') and r.probs is not None:  # Classification model
            probs = r.probs
            class_index = probs.top1
            class_name = r.names[class_index]
            confidence = float(probs.top1conf.cpu().numpy())
            
            # Make sure we're using a valid key from DISEASE_INFO
            if class_name in DISEASE_INFO[fruit_type]:
                disease_info = DISEASE_INFO[fruit_type][class_name]
            else:
                # Default to healthy if class_name not found
                disease_info = DISEASE_INFO[fruit_type]['healthy']
            
            disease_results.append({
                'name': class_name,
                'confidence': confidence,
                'info': disease_info
            })
        else:  # Detection model
            for box in r.boxes:
                class_index = int(box.cls[0])
                class_name = r.names[class_index]
                confidence = float(box.conf[0])
                
                # Make sure we're using a valid key from DISEASE_INFO
                if class_name in DISEASE_INFO[fruit_type]:
                    disease_info = DISEASE_INFO[fruit_type][class_name]
                else:
                    # Default to healthy if class_name not found
                    disease_info = DISEASE_INFO[fruit_type]['healthy']
                
                disease_results.append({
                    'name': class_name,
                    'confidence': confidence,
                    'info': disease_info
                })
    
    # If no disease was detected, return healthy
    if not disease_results:
        disease_results.append({
            'name': 'healthy',
            'confidence': 1.0,
            'info': DISEASE_INFO[fruit_type]['healthy']
        })
    
    return disease_results


def map_class_name(class_name):
    mapping = {
        'Banana Black Sigatoka Disease': 'black_sigatoka',
        'Banana Moko Disease': 'moko_disease',
        'Banana Yellow Sigatoka Disease': 'yellow_sigatoka',
        'Banana Panama Disease': 'panama_disease',
        'Banana Insect Pest Disease': 'insect_pest',
        'Banana Bract Mosaic Virus Disease': 'bract_mosaic',
        'Banana Healthy Leaf': 'healthy'
    }
    return mapping.get(class_name, 'healthy')


def detect_disease(model, image, fruit_type):
    """
    Detect diseases in fruits using trained YOLO models
    
    Parameters:
    model (YOLO): YOLOv8 model for disease detection
    image (PIL.Image): Image of the fruit
    fruit_type (str): Type of fruit ('banana', 'mango', or 'pomogranate')
    
    Returns:
    list: List of detected diseases with confidence scores
    """
    result = model(image)
    disease_results = []
    
    for r in result:
        if hasattr(r, 'probs') and r.probs is not None:  # Classification model
            probs = r.probs
            class_index = probs.top1
            class_name = r.names[class_index]
            confidence = float(probs.top1conf.cpu().numpy())
            disease_results.append({
                'name': class_name,
                'confidence': confidence,
                'info': DISEASE_INFO[fruit_type].get(class_name, DISEASE_INFO[fruit_type]['healthy'])
            })
        else:  # Detection model
            for box in r.boxes:
                class_index = int(box.cls[0])
                class_name = r.names[class_index]
                confidence = float(box.conf[0])
                mapped_name = map_class_name(class_name)
                disease_results.append({
                    'name': class_name,
                    'confidence': confidence,
                    'info': DISEASE_INFO[fruit_type].get(mapped_name, DISEASE_INFO[fruit_type]['healthy'])
                })
    
    # If no disease was detected, return healthy
    if not disease_results:
        disease_results.append({
            'name': 'healthy',
            'confidence': 1.0,
            'info': DISEASE_INFO[fruit_type]['healthy']
        })
        
    return disease_results


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def estimate_ripeness(image, fruit_type):
    """
    Estimate the ripeness percentage of a fruit based on color analysis
    
    Parameters:
    image (PIL.Image): Image of the fruit
    fruit_type (str): Type of fruit ('banana', 'mango', etc.)
    
    Returns:
    float: Estimated ripeness percentage (0-100)
    """
    # Convert PIL image to OpenCV format for color analysis
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if fruit_type == 'banana':
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range (ripe banana)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        # Define green color range (unripe banana)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        # Calculate yellow and green pixel ratios
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv_img, green_lower, green_upper)
        
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        yellow_pixels = np.sum(yellow_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        
        if (yellow_pixels + green_pixels) > 0:
            ripeness = (yellow_pixels / (yellow_pixels + green_pixels)) * 100
            return min(ripeness, 100)
        
    elif fruit_type == 'mango':
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define yellow/orange color range (ripe mango)
        ripe_lower = np.array([10, 100, 100])
        ripe_upper = np.array([30, 255, 255])
        
        # Define green color range (unripe mango)
        unripe_lower = np.array([35, 50, 50])
        unripe_upper = np.array([85, 255, 255])
        
        # Calculate ripe and unripe pixel ratios
        ripe_mask = cv2.inRange(hsv_img, ripe_lower, ripe_upper)
        unripe_mask = cv2.inRange(hsv_img, unripe_lower, unripe_upper)
        
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        ripe_pixels = np.sum(ripe_mask > 0)
        unripe_pixels = np.sum(unripe_mask > 0)
        
        if (ripe_pixels + unripe_pixels) > 0:
            ripeness = (ripe_pixels / (ripe_pixels + unripe_pixels)) * 100
            return min(ripeness, 100)
        
    elif fruit_type == 'pomogranate':
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges (ripe pomegranate)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Calculate red pixel ratio (pomegranates turn deep red when ripe)
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        red_pixels = np.sum(red_mask > 0)
        
        ripeness = (red_pixels / total_pixels) * 150  # Scale factor for pomegranate
        return min(ripeness, 100)
    
    # Default return if no specific analysis is available
    return 50  # Return 50% as a default value


def get_ripeness_category(ripeness_percentage):
    """
    Convert a ripeness percentage to a category
    
    Parameters:
    ripeness_percentage (float): Ripeness percentage (0-100)
    
    Returns:
    str: 'unripe' or 'ripe'
    """
    if ripeness_percentage < 60:
        return 'unripe'
    else:
        return 'ripe'



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




# Nutrition database for different fruits and ripeness levels
NUTRITION_INFO = {
    'banana': {
        'unripe': {
            'calories': 81,
            'carbs': '20g',
            'sugar': '6g',
            'fiber': '2.6g',
            'protein': '1g',
            'potassium': '358mg',
            'benefits': 'Contains resistant starch which acts as a prebiotic and supports gut health.'
        },
        'ripe': {
            'calories': 105,
            'carbs': '27g',
            'sugar': '14g',
            'fiber': '3g',
            'protein': '1.3g',
            'potassium': '422mg',
            'benefits': 'Higher in antioxidants compared to unripe bananas. Good energy source.'
        }
    },
    'mango': {
        'unripe': {
            'calories': 60,
            'carbs': '15g',
            'sugar': '8g',
            'fiber': '1.6g',
            'protein': '0.6g',
            'vitamin_c': '35mg',
            'benefits': 'Contains more vitamin C and less sugar than ripe mangoes.'
        },
        'ripe': {
            'calories': 99,
            'carbs': '24.7g',
            'sugar': '22.5g',
            'fiber': '2.6g',
            'protein': '1.4g',
            'vitamin_c': '60mg',
            'benefits': 'Rich in vitamins A and C, and antioxidants.'
        }
    },
    'pomogranate': {
        'unripe': {
            'calories': 68,
            'carbs': '17g',
            'sugar': '11g',
            'fiber': '3g',
            'protein': '1.5g',
            'vitamin_c': '9mg',
            'benefits': 'Higher in tannins which have astringent properties.'
        },
        'ripe': {
            'calories': 83,
            'carbs': '18.7g',
            'sugar': '13.7g',
            'fiber': '4g',
            'protein': '1.7g',
            'vitamin_c': '10.2mg',
            'benefits': 'Rich in antioxidants, particularly punicalagins which may help reduce inflammation.'
        }
    }
}


# Storage recommendations for different fruits and ripeness levels
STORAGE_RECOMMENDATIONS = {
    'banana': {
        'unripe': [
            'Store at room temperature (20-25°C/68-77°F).',
            'Keep away from direct sunlight.',
            'To speed up ripening, place in a paper bag with an apple or tomato (they release ethylene gas).',
            'Do not refrigerate unripe bananas as it stops the ripening process.'
        ],
        'ripe': [
            'Store at room temperature if consuming within 1-2 days.',
            'Refrigerate to extend shelf life (skin will darken but fruit remains fresh).',
            'For longer storage, peel and freeze in airtight containers or freezer bags.',
            'Separate from other fruits to prevent accelerated ripening.'
        ]
    },
    'mango': {
        'unripe': [
            'Store at room temperature (20-25°C/68-77°F).',
            'Place in a paper bag with a banana or apple to speed up ripening.',
            'Keep away from direct sunlight.',
            'Check daily for ripeness by gently pressing the fruit.'
        ],
        'ripe': [
            'Store in the refrigerator for up to 5 days.',
            'Bring to room temperature before eating for best flavor.',
            'To store cut mango, place in an airtight container in the refrigerator for 2-3 days.',
            'For longer storage, cut into pieces and freeze on a tray, then transfer to freezer bags.'
        ]
    },
    'pomogranate': {
        'unripe': [
            'Store at room temperature away from direct sunlight.',
            'Check regularly for ripeness.',
            'Pomegranates generally do not continue to ripen once harvested, so its best to purchase them ripe.'
        ],
        'ripe': [
            'Store whole pomegranates in a cool, dry place for up to 1 month.',
            'Refrigerate whole pomegranates for up to 2 months.',
            'Store extracted seeds (arils) in an airtight container in the refrigerator for up to 5 days.',
            'Freeze arils in a single layer on a tray, then transfer to freezer bags for up to 3 months.'
        ]
    }
}


DISEASE_INFO = {
    'banana': {
        'healthy': {
            'description': 'No disease detected. The banana appears healthy.',
            'treatment': 'Continue with regular care and monitoring.',
            'prevention': 'Maintain good air circulation and avoid excess moisture.'
        },
        'black_sigatoka': {
            'description': 'Black Sigatoka is a fungal disease causing black leaf spots that eventually merge.',
            'treatment': 'Remove infected leaves. Apply fungicides as recommended by agricultural experts.',
            'prevention': 'Use resistant varieties. Ensure proper spacing for air circulation.'
        },
        'crown_rot': {
            'description': 'Crown rot affects the crown of the banana hand, showing water-soaked areas that turn brown.',
            'treatment': 'Post-harvest treatment with fungicides. Prompt cooling after harvest.',
            'prevention': 'Careful handling during harvest. Maintain clean harvesting tools.'
        },
        'panama_disease': {
            'description': 'Panama disease (Fusarium wilt) causes yellowing and wilting of leaves, starting from the oldest.',
            'treatment': 'No effective treatment exists. Remove and destroy infected plants.',
            'prevention': 'Use resistant varieties. Avoid introducing infected soil or planting material.'
        }
    },
    'mango': {
        'healthy': {
            'description': 'No disease detected. The mango appears healthy.',
            'treatment': 'Continue with regular care and monitoring.',
            'prevention': 'Maintain good orchard sanitation and proper pruning.'
        },
        'anthracnose': {
            'description': 'Anthracnose causes dark spots on fruits, flowers, and leaves that eventually expand.',
            'treatment': 'Apply copper-based fungicides. Prune infected parts.',
            'prevention': 'Regular fungicide sprays during flowering and fruit set. Maintain tree spacing.'
        },
        'powdery_mildew': {
            'description': 'Powdery mildew appears as white powdery patches on leaves, shoots, and young fruits.',
            'treatment': 'Apply sulfur-based fungicides. Remove severely infected parts.',
            'prevention': 'Ensure good air circulation. Avoid overhead irrigation.'
        },
        'bacterial_canker': {
            'description': 'Bacterial canker shows as raised lesions on stems and dark spots on fruits with gummy ooze.',
            'treatment': 'Prune infected branches. Apply copper-based bactericides.',
            'prevention': 'Use disease-free planting material. Avoid wounding trees.'
        }
    },
    'pomogranate': {
        'healthy': {
            'description': 'No disease detected. The pomegranate appears healthy.',
            'treatment': 'Continue with regular care and monitoring.',
            'prevention': 'Maintain proper pruning and orchard sanitation.'
        },
        'fruit_rot': {
            'description': 'Fruit rot appears as soft, water-soaked areas that become covered with fungal growth.',
            'treatment': 'Remove infected fruits. Apply fungicides as recommended.',
            'prevention': 'Avoid fruit injuries. Ensure proper drainage in orchards.'
        },
        'bacterial_blight': {
            'description': 'Bacterial blight causes water-soaked lesions on leaves and fruits that turn dark brown.',
            'treatment': 'Prune infected parts. Apply copper-based bactericides.',
            'prevention': 'Use disease-free planting material. Avoid overhead irrigation.'
        },
        'leaf_spot': {
            'description': 'Leaf spot disease shows as circular spots on leaves that may have dark margins.',
            'treatment': 'Apply fungicides. Remove severely infected leaves.',
            'prevention': 'Ensure good air circulation. Maintain proper spacing.'
        }
    }
}


if __name__ == '__main__':
    check_templates()
    app.run(host="0.0.0.0", debug=True)
