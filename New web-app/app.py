# -*- coding: utf-8 -*-
from flask import Flask, render_template, send_from_directory, Response, jsonify, request, url_for
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import time
from collections import Counter
import logging
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ======================
# Configuration
# ======================
MODEL_PATH = r"C:\Drive F\Code\Project\Indian-Sign-Language-To-Multiple-Regional-Language-Conversion\src\fFnal_training_Results\weights\best.pt"
GEMINI_API_KEY = "AIzaSyDCGfIOe2f6Un0QX7WHeLL3Nwp652kMxk4"
# Optimized Smooth Detection Settings
TARGET_FPS = 15           # Video display frame rate (increased from 10)
DETECTION_INTERVAL = 0.2  # Detection processing interval (reduced from 0.3)
ANALYSIS_INTERVAL = 0.8   # Sign analysis interval (reduced from 1.0)
# Add to configuration
DEFAULT_STATIC_IMAGE = r"C:\Drive F\Code\Project\IWeb-app-Indian-Sign-Language-To-Multiple-Regional-Language-Conversion - Copy\New web-app\static\impact-of-ai-sign-language-translators.jpg"  # Add this path


# ======================
# Translation Map
# ======================
TRANSLATION_MAP = {
    'en': {
        '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
        '6': '6', '7': '7', '8': '8', '9': '9',
        'Band Aid': 'Band Aid', 'Bye': 'Bye', 'Cough': 'Cough',
        'Eat': 'Eat', 'Home': 'Home', 'I': 'I', 'Like': 'Like',
        'Love': 'Love', 'Request': 'Request', 'Stop': 'Stop',
        'What': 'What', 'When': 'When'
    },
    'hi': {
        '1': '१', '2': '२', '3': '३', '4': '४', '5': '५',
        '6': '६', '7': '७', '8': '८', '9': '९',
        'Band Aid': 'बैंड एड', 'Bye': 'अलविदा', 'Cough': 'खांसी',
        'Eat': 'खाना', 'Home': 'घर', 'I': 'मैं', 'Like': 'पसंद',
        'Love': 'प्यार', 'Request': 'निवेदन', 'Stop': 'रुकें',
        'What': 'क्या', 'When': 'कब'
    },
    'mr': {
        '1': '१', '2': '२', '3': '३', '4': '४', '5': '५',
        '6': '६', '7': '७', '8': '८', '9': '९',
        'Band Aid': 'बॅंड एड', 'Bye': 'निरोप', 'Cough': 'खोकला',
        'Eat': 'खाणे', 'Home': 'घर', 'I': 'मी', 'Like': 'आवड',
        'Love': 'प्रेम', 'Request': 'विनंती', 'Stop': 'थांबा',
        'What': 'काय', 'When': 'केव्हा'
    },
    'pa': {
        '1': '੧', '2': '੨', '3': '੩', '4': '੪', '5': '੫',
        '6': '੬', '7': '੭', '8': '੮', '9': '੯',
        'Band Aid': 'ਬੈਂਡ ਐਡ', 'Bye': 'ਅਲਵਿਦਾ', 'Cough': 'ਖੰਘ',
        'Eat': 'ਖਾਣਾ', 'Home': 'ਘਰ', 'I': 'ਮੈਂ', 'Like': 'ਪਸੰਦ',
        'Love': 'ਪਿਆਰ', 'Request': 'ਬੇਨਤੀ', 'Stop': 'ਰੁਕੋ',
        'What': 'ਕੀ', 'When': 'ਕਦੋਂ'
    },
    'ta': {
        '1': '௧', '2': '௨', '3': '௩', '4': '௪', '5': '௫',
        '6': '௬', '7': '௭', '8': '௮', '9': '௯',
        'Band Aid': 'கட்டு மருந்து', 'Bye': 'சென்று வருகிறேன்', 
        'Cough': 'இருமல்', 'Eat': 'சாப்பிடு', 'Home': 'வீடு', 
        'I': 'நான்', 'Like': 'விரும்பு', 'Love': 'காதல்',
        'Request': 'கோரிக்கை', 'Stop': 'நிறுத்து', 'What': 'என்ன',
        'When': 'எப்போது'
    }
}
# Thread-safe state management
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detected_signs = []
        self.gesture_buffer = []
        self.last_analysis = time.time()
        self.last_detection = time.time()
        self.current_frame = None
        self.is_processing = False
        self.detection_active = False  # Add detection state flag

state = AppState()
# ======================
# Initialization
# ======================
try:
    model = YOLO(MODEL_PATH)
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    exit(1)


def generate_frames():
    cap = None
    try:
        while True:
            if state.detection_active:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                success, frame = cap.read()
                if not success:
                    time.sleep(0.1)
                    continue
                
                # Detection processing logic (same as before)
                if (time.time() - state.last_detection > DETECTION_INTERVAL and 
                    not state.is_processing):
                    
                    state.is_processing = True
                    try:
                        results = model(frame, verbose=False)
                        annotated_frame = results[0].plot()
                        
                        with state.lock:
                            state.current_frame = annotated_frame.copy()
                            state.last_detection = time.time()
                            
                            current_signs = [model.names[int(box.cls)] for r in results for box in r.boxes]
                            state.gesture_buffer.extend(current_signs)
                            
                            if time.time() - state.last_analysis > ANALYSIS_INTERVAL:
                                if state.gesture_buffer:
                                    most_common = Counter(state.gesture_buffer).most_common(1)[0][0]
                                    if (not state.detected_signs or 
                                        state.detected_signs[-1] != most_common):
                                        state.detected_signs.append(most_common)
                                        state.detected_signs = state.detected_signs[-5:]
                                state.gesture_buffer = []
                                state.last_analysis = time.time()
                    finally:
                        state.is_processing = False
                else:
                    with state.lock:
                        annotated_frame = state.current_frame if state.current_frame is not None else frame
            else:
                # Show placeholder when detection is inactive
                if cap is not None:
                    cap.release()
                    cap = None
                annotated_frame = cv2.imread(DEFAULT_STATIC_IMAGE)
                if annotated_frame is None:
                    annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(annotated_frame, "Camera Inactive", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except Exception as e:
        logging.error(f"Video feed error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()


# ======================
# API Endpoints (Improved Error Handling)
# ======================


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    try:
        with state.lock:
            state.detection_active = not state.detection_active
            if not state.detection_active:  # Clear detections when stopping
                state.detected_signs = []
                state.gesture_buffer = []
            return jsonify({
                "status": "success",
                "detection_active": state.detection_active
            })
    except Exception as e:
        logging.error(f"Toggle detection error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process', methods=['GET'])
def process():
    try:
        with state.lock:
            return jsonify({
                "sequence": state.detected_signs,
                "status": "success"
            })
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        signs = data.get('signs', [])
        lang = data.get('lang', 'en')
        
        if not signs:
            return jsonify({"error": "No signs provided"}), 400
        
        # Always translate using the mapping first
        translated_words = [
            TRANSLATION_MAP.get(lang, {}).get(sign, sign)
            for sign in signs
        ]
        
        # Always use Gemini for sentence construction
        prompt = f"""
          Detected gesture sequence: {translated_words}
            Construct a coherent {lang} sentence that a person might communicate using these gestures.

            Instructions:
            1. If the gesture sequence matches or closely relates to any of the following examples, respond with the corresponding sentence:
                - "What Eat" → "What should I eat?"
                - "When Eat" → "When will we eat?"
                - "What Love" → "What do you love?"
                - "When Bye" → "When should I say goodbye?"
                - "What Request" → "What is the request?"
                - "I Like Home" → "I like home."
                - "What Eat Cough" → "What to eat in cough?"
                - "What Eat" → "What to eat?"
                - "I Request Band Aid" → "I request to do me Band Aid."
                - "Request Band Aid Cough" → "Request medicine for cough."
                - "When Eat Cough Stop" → "When I eat well, cough stopped!"
                - "I Request Stop Home" → "I request you to stop at home."
                - "I Love Work" → "I love to work."
                - "I Home Bye" → "I am going home, bye."
                - "I Love Home" → "I love home."
                - "I Request Stop" → "I request you to stop."
            
            2. If no close match is found, create a new meaningful and professional sentence based on the gesture sequence.
            3. Keep the sentence simple and correct according to {lang} grammar.
            4. Respond only with the sentence in {lang}.
        """

        response = gemini_model.generate_content(prompt)
        translation = response.text

        # Additional verification for non-English
        if lang != 'en':
            if all(ord(c) < 128 for c in translation):
                prompt += "\nCRITICAL: Respond using native script for {lang} language, NOT Roman characters."
                response = gemini_model.generate_content(prompt)
                translation = response.text

        return jsonify({
            "translation": translation,
            "status": "success",
            "method": "gemini"
        })
        
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        with state.lock:
            state.detected_signs = []
            state.gesture_buffer = []
            state.last_analysis = time.time()
        return jsonify({
            "status": "reset",
            "message": "Detection history cleared"
        })
    except Exception as e:
        logging.error(f"Reset error: {str(e)}")
        return jsonify({"error": str(e)}), 500



# ======================
# Application Routes (Keep All Pages)
# ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/core_app')
def core_app():
    return render_template('core_app.html')

@app.route('/yolo')
def yolo():
    return render_template('yolo.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/llm')
def llm():
    return render_template('llm.html')

@app.route('/api')
def api():
    return render_template('api.html')
# ======================
# Static Files & Errors
# ======================

@app.route('/static/demos/<path:filename>')
def demo_files(filename):
    return send_from_directory('static/demos', filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)