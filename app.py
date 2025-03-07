# -*- coding: utf-8 -*-
import sys
import cv2
import time
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QComboBox, 
                             QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
                             QGroupBox, QFrame, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase
import google.generativeai as genai

# Constants
FPS = 2
MAX_SIGNS = 5
ANALYSIS_DURATION = 1
MODEL_PATH = r"C:\Drive F\Code\Project\Indian-Sign-Language-To-Multiple-Regional-Language-Conversion\src\fFnal_training_Results\weights\best.pt"
GEMINI_API_KEY = "AIzaSyDwqyUxpo7sy-1y1SgfXQtc2FwEIR9VdP8"

# Font Paths for Different Languages
FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"
}

# Translation Map
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

LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'ta': 'Tamil'
}

class GeminiThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, model):
        super().__init__()
        self.prompt = prompt
        self.model = model

    def run(self):
        try:
            response = self.model.generate_content(self.prompt)
            self.finished_signal.emit(response.text)
        except Exception as e:
            self.error_signal.emit(f"Error generating sentence: {str(e)}")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_signs_signal = pyqtSignal(list)

    def __init__(self, model_path, font_paths, translation_map):
        super().__init__()
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        self.detected_signs = []
        self.gesture_buffer = []
        self.start_time = time.time()
        self.lang_code = 'en'
        self.font_paths = font_paths
        self.translation_map = translation_map
        self.running = True

    def clear_history(self):
        self.detected_signs = []
        self.gesture_buffer = []
        self.update_signs_signal.emit([])  # Explicitly send empty update

    # It will reset the sign after displaying Sentence.
    def reset_detected_signs(self):
        self.detected_signs = []
        self.gesture_buffer = []

    def run(self):
        while self.running:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame)
            annotated_frame = self.draw_annotations(frame, results)
            self.change_pixmap_signal.emit(annotated_frame)

            current_gestures = [
                self.model.names[int(cls)]
                for result in results
                for cls in result.boxes.cls.cpu().numpy()
                if result.boxes is not None
            ]
            self.gesture_buffer.extend(current_gestures)
            self.gesture_buffer, self.start_time = self.update_gesture_list()

            elapsed = time.time() - frame_start
            if elapsed < 1/FPS:
                time.sleep((1/FPS) - elapsed)

    def draw_annotations(self, frame, results):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(self.font_paths[self.lang_code], 30)
        except (IOError, KeyError):
            font = ImageFont.load_default()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                original_label = self.model.names[int(cls)]
                translated_label = self.translation_map[self.lang_code].get(original_label, original_label)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 30), translated_label, fill="green", font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def update_gesture_list(self):
        current_time = time.time()
        if current_time - self.start_time >= ANALYSIS_DURATION:
            if self.gesture_buffer:
                most_common = Counter(self.gesture_buffer).most_common(1)[0][0]
                if not self.detected_signs or self.detected_signs[-1] != most_common:
                    self.detected_signs.append(most_common)
                    self.detected_signs = self.detected_signs[-MAX_SIGNS:]
                    self.update_signs_signal.emit(self.detected_signs.copy())
            return [], current_time
        return self.gesture_buffer, self.start_time

    def set_language(self, lang_code):
        self.lang_code = lang_code

    def stop(self):
        self.running = False
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Detection")
        self.setGeometry(100, 100, 1366, 768)
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")
        self.is_translating = False

        # Color Palette
        self.COLORS = {
            'primary': '#2A3F54',
            'secondary': '#1ABB9C',
            'accent': '#3498DB',
            'background': '#F7F9FC',
            'text': '#2C3E50',
            'success': '#27AE60',
            'warning': '#F1C40F',
            'danger': '#E74C3C'
        }

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # Header
        header = QLabel("Sign Language Interpreter")
        header.setStyleSheet(f"""
            background-color: {self.COLORS['primary']};
            color: white;
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            border-radius: 10px;
        """)
        header.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(header)

        # Main Content
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)

        # Video Panel
        video_panel = QFrame()
        video_panel.setStyleSheet(f"""
            background-color: {self.COLORS['background']};
            border-radius: 10px;
            border: 2px solid {self.COLORS['primary']};
        """)
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        video_layout.addWidget(self.video_label)
        content_layout.addWidget(video_panel, 70)

        # Control Panel
        control_panel = QFrame()
        control_panel.setStyleSheet(f"""
            background-color: white;
            border-radius: 10px;
            border: 2px solid {self.COLORS['primary']};
        """)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)

        # Detection Status
        self.status_indicator = QLabel("● Ready")
        self.status_indicator.setStyleSheet(f"""
            color: {self.COLORS['success']};
            font-size: 16px;
            font-weight: bold;
        """)
        control_layout.addWidget(self.status_indicator)

        # Detection History
        self.signs_panel = QGroupBox("Translation Panel")
        self.signs_panel.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {self.COLORS['accent']};
                border-radius: 8px;
                margin-top: 10px;
                color: {self.COLORS['text']};
                font-size: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        signs_layout = QVBoxLayout(self.signs_panel)
        
        self.signs_display = QLabel("Detected signs will appear here...")
        self.signs_display.setStyleSheet(f"""
            background-color: {self.COLORS['background']};
            color: {self.COLORS['text']};
            border-radius: 5px;
            padding: 15px;
            font-size: 16px;
            min-height: 100px;
        """)
        self.signs_display.setAlignment(Qt.AlignCenter)
        self.signs_display.setWordWrap(True)
        signs_layout.addWidget(self.signs_display)
        control_layout.addWidget(self.signs_panel)

        # Controls
        controls_group = QGroupBox("Settings")
        controls_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {self.COLORS['secondary']};
                border-radius: 8px;
                margin-top: 10px;
                color: {self.COLORS['text']};
                font-size: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(15)

        # Language Selection
        lang_widget = QWidget()
        lang_layout = QHBoxLayout(lang_widget)
        lang_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Hindi", "Marathi", "Punjabi", "Tamil"])
        self.lang_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: white;
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['accent']};
                border-radius: 5px;
                padding: 8px;
            }}
        """)
        lang_layout.addWidget(self.lang_combo)
        self.lang_combo.currentTextChanged.connect(self.update_selected_language)
        controls_layout.addWidget(lang_widget)

        # Action Buttons
        self.translate_btn = QPushButton("Translate Sentence")
        self.translate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLORS['accent']};
                color: white;
                border-radius: 5px;
                padding: 12px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.COLORS['secondary']};
            }}
        """)
        self.translate_btn.clicked.connect(self.translate_sentence)
        controls_layout.addWidget(self.translate_btn)

        self.quit_btn = QPushButton("Exit Application")
        self.quit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLORS['danger']};
                color: white;
                border-radius: 5px;
                padding: 12px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #C44133;
            }}
        """)
        self.quit_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.quit_btn)

        control_layout.addWidget(controls_group)
        content_layout.addWidget(control_panel, 30)
        self.main_layout.addWidget(content_widget)

        # Thread Setup
        self.thread = VideoThread(MODEL_PATH, FONT_PATHS, TRANSLATION_MAP)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_signs_signal.connect(self.update_detected_signs)
        self.thread.start()

        # Font Setup
        self.font_ids = {}
        for lang, path in FONT_PATHS.items():
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                self.font_ids[lang] = font_id

        self.detected_signs = []
        self.current_lang = 'en'

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def update_selected_language(self, lang_name):
        lang_map = {
            "English": "en", 
            "Hindi": "hi",
            "Marathi": "mr",
            "Punjabi": "pa",
            "Tamil": "ta"
        }
        self.current_lang = lang_map.get(lang_name, "en")
        self.thread.set_language(self.current_lang)
        
        # Reset detected signs in both thread and UI
        self.thread.reset_detected_signs()  # Clear thread's history
        self.detected_signs = []  # Clear local history
        self.update_detected_signs(self.detected_signs)  # Update GUI
        
        # Update display font
        if self.current_lang in self.font_ids:
            families = QFontDatabase.applicationFontFamilies(self.font_ids[self.current_lang])
            if families:
                font = QFont(families[0], 16)
                self.signs_display.setFont(font)

    def update_detected_signs(self, signs):
        if self.is_translating:
            return  # Ignore updates during translation
        
        self.detected_signs = signs.copy()
        translated = [
            TRANSLATION_MAP[self.current_lang].get(s, s)
            for s in self.detected_signs
        ]
        display_text = "\n".join(translated) if translated else "Waiting for signs..."
        self.signs_display.setText(display_text)

    # Modified translation method
    def translate_sentence(self):
        if not self.detected_signs:
            self.signs_display.setText(TRANSLATION_MAP[self.current_lang].get('No signs detected', 'No signs detected'))
            QTimer.singleShot(2000, lambda: self.signs_display.setText("Waiting for signs..."))
            return
        
        self.is_translating = True
        sentence = ' '.join(self.detected_signs)
        lang_name = LANGUAGE_NAMES.get(self.current_lang, 'English')
        
        # Clear previous detections immediately
        self.thread.clear_history()
        self.detected_signs = []
        
        
        prompt = f"""
            Detected gesture sequence: {sentence}
            Construct a coherent {lang_name} sentence that a person might communicate using these gestures.

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
            3. Keep the sentence simple and correct according to {lang_name} grammar.
            4. Respond only with the sentence in {lang_name}.
            """

        

        
        self.signs_display.setText(TRANSLATION_MAP[self.current_lang].get('Processing...', 'Processing...'))
        self.gemini_thread = GeminiThread(prompt, self.genai_model)
        self.gemini_thread.finished_signal.connect(self.handle_translation_result)
        self.gemini_thread.error_signal.connect(self.handle_translation_error)
        self.gemini_thread.start()


    def handle_translation_result(self, result):
        self.signs_display.setText(result)
        QTimer.singleShot(5000, self.revert_display)

    def handle_translation_error(self, error_msg):
        self.signs_display.setText(error_msg)
        QTimer.singleShot(5000, self.revert_display)

    def revert_display(self):
        self.is_translating = False
        # Clear both thread and local history
        self.thread.clear_history()
        self.detected_signs = []
        # Force immediate UI update
        self.signs_display.setText("Waiting for signs...")

    def closeEvent(self, event):
        self.thread.stop()
        if hasattr(self, 'gemini_thread') and self.gemini_thread.isRunning():
            self.gemini_thread.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())