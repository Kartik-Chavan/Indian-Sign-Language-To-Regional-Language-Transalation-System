# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Constants
FPS = 2
MAX_SIGNS = 5
DISPLAY_TIME = 2
ANALYSIS_DURATION = 1
MODEL_PATH = r"C:\Drive F\Code\Project\Indian-Sign-Language-To-Multiple-Regional-Language-Conversion\src\fFnal_training_Results\weights\best.pt"

# Font Paths for Different Languages
FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"
}
CORE_SIGNS = [
    # Medical Essentials
    "Cough", "Headache", "Fever", "Band Aid",
    
    # General Fundamentals
    "What", "When", "Request", "Stop", 
    
    # Daily Life
    "Eat", "Love", "Like",
    
    # Locations
    "Home", "School",
    
    # Identity
    "I",
    
    # Social
    "Bye",
    
    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "8", "9"
]
TRANSLATION_MAP = {
    'en': {'C': 'C', 'Friends': 'Friends', 'H': 'H', 'Hello': 'Hello', 'L': 'L', 'O': 'O', 'Please': 'Please', 'Thanks': 'Thanks'},
    'hi': {'C': 'सी', 'Friends': 'दोस्त', 'H': 'एच', 'Hello': 'नमस्ते', 'L': 'एल', 'O': 'ओ', 'Please': 'कृपया', 'Thanks': 'धन्यवाद'},
    'mr': {'C': 'सी', 'Friends': 'मित्र', 'H': 'एच', 'Hello': 'नमस्कार', 'L': 'एल', 'O': 'ओ', 'Please': 'कृपया', 'Thanks': 'धन्यवाद'},
    'pa': {'C': 'ਸੀ', 'Friends': 'ਦੋਸਤ', 'H': 'ਐਚ', 'Hello': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'L': 'ਐਲ', 'O': 'ਓ', 'Please': 'ਕ੍ਰਿਪਾ ਕਰਕੇ', 'Thanks': 'ਧੰਨਵਾਦ'},
    'ta': {'C': 'சி', 'Friends': 'நண்பர்கள்', 'H': 'எச்', 'Hello': 'வணக்கம்', 'L': 'எல்', 'O': 'ஓ', 'Please': 'தயவுசெய்து', 'Thanks': 'நன்றி'}
}

def select_language():
    """Prompt user to select a language and return the language code."""
    print("\nAvailable Languages:")
    print("1. English\n2. Hindi\n3. Marathi\n4. Punjabi\n5. Tamil")
    lang_choice = input("Enter choice (1-5): ").strip()
    
    lang_code = {
        '1': 'en', '2': 'hi', '3': 'mr', 
        '4': 'pa', '5': 'ta'
    }.get(lang_choice, 'en')
    
    print(f"Selected language: {lang_code}")
    return lang_code

def initialize_model_and_camera():
    """Initialize YOLO model and webcam"""
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    return model, cap

def process_frame(model, frame):
    """Process a frame and return detection results"""
    return model(frame)

def draw_annotations(frame, results, model, lang_code):
    """Draw bounding boxes and translated labels with dynamic fonts."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(FONT_PATHS[lang_code], 30)
    except (IOError, KeyError):
        font = ImageFont.load_default()
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            original_label = model.names[int(cls)]
            translated_label = TRANSLATION_MAP[lang_code].get(original_label, original_label)
            
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 30), translated_label, fill="green", font=font)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def update_gesture_list(gesture_buffer, detected_signs, analysis_duration, max_signs, start_time):
    """Update detected signs list with time-based analysis"""
    if time.time() - start_time >= analysis_duration:
        if gesture_buffer:
            most_common = Counter(gesture_buffer).most_common(1)[0][0]
            if not detected_signs or detected_signs[-1] != most_common:
                detected_signs.append(most_common)
                detected_signs = detected_signs[-max_signs:]
        return [], time.time()
    return gesture_buffer, start_time

def handle_user_input(frame, detected_signs, display_time, lang_code):
    """Handle all user inputs including language change"""
    key = cv2.waitKey(1) & 0xFF
    new_lang = lang_code
    
    if key == ord('e'):
        # Display detected signs
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype(FONT_PATHS[lang_code], 30)
        except (IOError, KeyError):
            font = ImageFont.load_default()
        
        translated_signs = [TRANSLATION_MAP[lang_code].get(s, s) for s in detected_signs]
        display_text = ", ".join(translated_signs)
        
        draw.text((20, 50), display_text, fill="green", font=font)
        display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('Detected Signs', display_frame)
        cv2.waitKey(DISPLAY_TIME * 1200)
        detected_signs.clear()
        gesture_buffer = []
        cv2.destroyWindow('Detected Signs')
    
    elif key == ord('c'):
        # Change language
        new_lang = select_language()
        detected_signs.clear()
        gesture_buffer = []
        print("Language changed. Resetting detected signs.")
    
    return key, new_lang

def enforce_fps(frame_start_time, fps):
    """Maintain specified FPS"""
    elapsed = time.time() - frame_start_time
    if elapsed < 1/fps:
        time.sleep((1/fps) - elapsed)

def main():
    """Main application loop"""
    model, cap = initialize_model_and_camera()
    lang_code = select_language()
    
    detected_signs = []
    gesture_buffer = []
    start_time = time.time()

    print("\nControls:")
    print("Press 'E' - Show detected signs")
    print("Press 'C' - Change language")
    print("Press 'Q' - Quit\n")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = process_frame(model, frame)
        annotated_frame = draw_annotations(frame, results, model, lang_code)
        cv2.imshow('Sign Language Detection', annotated_frame)

        # Update gestures
        current_gestures = [
            model.names[int(cls)] 
            for result in results 
            for cls in result.boxes.cls.cpu().numpy() 
            if result.boxes is not None
        ]
        gesture_buffer.extend(current_gestures)
        gesture_buffer, start_time = update_gesture_list(
            gesture_buffer, detected_signs, 
            ANALYSIS_DURATION, MAX_SIGNS, start_time
        )

        # Handle input and language change
        key, new_lang = handle_user_input(annotated_frame, detected_signs, DISPLAY_TIME, lang_code)
        if new_lang != lang_code:
            lang_code = new_lang
        if key == ord('q'):
            break
            
        enforce_fps(frame_start, FPS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()