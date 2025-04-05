from flask import Flask, render_template, send_from_directory, Response, jsonify, request, url_for
import os
import cv2
import numpy as np
import threading
from ultralytics import YOLO
from collections import Counter
import time
import logging
import google.generativeai as genai

app = Flask(__name__)

# ======================
# Configuration
# ======================
MODEL_PATH = "best.pt"
DEFAULT_STATIC_IMAGE = os.path.abspath(os.path.join("static", "demos", "placeholder.jpg"))
GEMINI_API_KEY = "AIzaSyDCGfIOe2f6Un0QX7WHeLL3Nwp652kMxk4"
TARGET_FPS = 15
DETECTION_INTERVAL = 0.2
ANALYSIS_INTERVAL = 0.8

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
    # ... [Include all other language mappings]
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
# ======================
# Initialization
# ======================
if not os.path.exists(DEFAULT_STATIC_IMAGE):
    raise FileNotFoundError(f"Default image missing: {DEFAULT_STATIC_IMAGE}")

try:
    model = YOLO(MODEL_PATH)
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logging.info("✅ Models loaded successfully")
except Exception as e:
    logging.critical(f"❌ Model loading failed: {str(e)}")
    exit(1)

# ======================
# State Management (Improved)
# ======================
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detected_signs = []
        self.gesture_buffer = []
        self.last_analysis = time.time()
        self.last_detection = time.time()
        self.current_frame = None
        self.is_processing = False
        self.detection_active = False
        self.cap = None
        self.static_image = cv2.imread(DEFAULT_STATIC_IMAGE)  # Pre-load static image

state = AppState()

def create_status_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def generate_frames():
    while True:
        try:
            if not state.detection_active:
                if state.cap and state.cap.isOpened():
                    state.cap.release()
                    state.cap = None
                
                # Use pre-loaded static image
                frame = state.static_image.copy() if state.static_image is not None \
                    else create_status_frame("System Ready")
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            if state.cap is None or not state.cap.isOpened():
                state.cap = cv2.VideoCapture(0)
                if not state.cap.isOpened():
                    error_frame = create_status_frame("Camera Error")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(1)
                    continue
                
                state.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            success, frame = state.cap.read()
            if not success:
                error_frame = create_status_frame("Frame Read Error")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            current_time = time.time()
            if (current_time - state.last_detection > DETECTION_INTERVAL and 
                not state.is_processing):
                
                state.is_processing = True
                try:
                    results = model(frame, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    with state.lock:
                        state.current_frame = annotated_frame
                        state.last_detection = current_time
                        
                        current_signs = [model.names[int(box.cls)] for r in results for box in r.boxes]
                        state.gesture_buffer.extend(current_signs)
                        
                        if current_time - state.last_analysis > ANALYSIS_INTERVAL:
                            if state.gesture_buffer:
                                most_common = Counter(state.gesture_buffer).most_common(1)[0][0]
                                if (not state.detected_signs or 
                                    state.detected_signs[-1] != most_common):
                                    state.detected_signs.append(most_common)
                                    state.detected_signs = state.detected_signs[-5:]
                            state.gesture_buffer = []
                            state.last_analysis = current_time
                except Exception as e:
                    logging.error(f"Detection error: {str(e)}")
                    annotated_frame = create_status_frame("Processing Error")
                finally:
                    state.is_processing = False
            else:
                with state.lock:
                    annotated_frame = state.current_frame if state.current_frame is not None else frame

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logging.error(f"Video stream error: {str(e)}")
            error_frame = create_status_frame("System Error")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

# ======================
# API Endpoints (Improved Error Handling)
# ======================
@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    try:
        with state.lock:
            state.detection_active = not state.detection_active
            if not state.detection_active:
                state.detected_signs = []
                state.gesture_buffer = []
                if state.cap and state.cap.isOpened():
                    state.cap.release()
                    state.cap = None
            
            return jsonify({
                "status": "success",
                "detection_active": state.detection_active
            })
    except Exception as e:
        logging.error(f"Toggle error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        signs = data.get('signs', [])
        lang = data.get('lang', 'en')
        
        if not signs:
            return jsonify({"error": "No signs provided"}), 400
        
        translated_words = [
            TRANSLATION_MAP.get(lang, {}).get(sign, sign)
            for sign in signs
        ]
        
        prompt = f"""Detected gesture sequence: {translated_words}
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
        if not response.parts:
            raise ValueError("Empty response from Gemini API")
            
        translation = response.text

        if lang != 'en' and all(ord(c) < 128 for c in translation):
            prompt += "\nCRITICAL: Respond in native script, not Roman characters."
            response = gemini_model.generate_content(prompt)
            translation = response.text

        return jsonify({
            "translation": translation,
            "status": "success"
        })
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        if hasattr(response, 'prompt_feedback'):
            logging.error(f"Gemini feedback: {response.prompt_feedback}")
        return jsonify({"error": str(e)}), 500

# ======================
# Application Routes (Keep All Pages)
# ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html',
                         video_feed_url=url_for('video_feed'),
                         toggle_url=url_for('toggle_detection'),
                         process_url=url_for('process'),
                         translate_url=url_for('translate'),
                         reset_url=url_for('reset'))

@app.route('/yolo')
def yolo():
    return render_template('yolo.html')

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
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)