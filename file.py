'''# -*- coding: utf-8 -*-
import sys
import cv2
import time
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QComboBox, 
                             QPushButton, QVBoxLayout, QWidget, QDialog)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Constants
FPS = 2
MAX_SIGNS = 5
ANALYSIS_DURATION = 1
MODEL_PATH = 'Kartik_model_file.pt'

# Font Paths for Different Languages
FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"
}

# Updated Translation Map
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


# Video Processing Thread
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

        # Real-time feedback
        if self.detected_signs:
            recent_sign = self.detected_signs[-1]
            translated_recent = self.translation_map[self.lang_code].get(recent_sign, recent_sign)
            draw.text((20, frame.shape[0] - 50), f"Recent: {translated_recent}", fill="blue", font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def update_gesture_list(self):
        if time.time() - self.start_time >= ANALYSIS_DURATION:
            if self.gesture_buffer:
                most_common = Counter(self.gesture_buffer).most_common(1)[0][0]
                if not self.detected_signs or self.detected_signs[-1] != most_common:
                    self.detected_signs.append(most_common)
                    self.detected_signs = self.detected_signs[-MAX_SIGNS:]
                    self.update_signs_signal.emit(self.detected_signs)
            return [], time.time()
        return self.gesture_buffer, self.start_time

    def set_language(self, lang_code):
        self.lang_code = lang_code

    def stop(self):
        self.running = False
        self.cap.release()

# Dialog for Detected Signs
class SignsDialog(QDialog):
    def __init__(self, signs, lang_code, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detected Signs")
        layout = QVBoxLayout(self)
        label = QLabel()
        if signs:
            translated_signs = [TRANSLATION_MAP[lang_code].get(s, s) for s in signs]
            display_text = ", ".join(translated_signs)
        else:
            display_text = "No signs detected yet."
        label.setText(display_text)
        layout.addWidget(label)
        self.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        label.setStyleSheet("font-size: 16px; color: #333;")

# Main Application Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Detection")
        self.setGeometry(100, 100, 800, 600)

        # Central Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video Display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Language Selection
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Hindi", "Marathi", "Punjabi", "Tamil"])
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        self.layout.addWidget(self.lang_combo)

        # Buttons
        self.show_signs_btn = QPushButton("Show Detected Signs")
        self.show_signs_btn.clicked.connect(self.show_detected_signs)
        self.layout.addWidget(self.show_signs_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        self.layout.addWidget(self.quit_btn)

        # Styling
        self.setStyleSheet("background-color: #e0e0e0;")
        self.lang_combo.setStyleSheet("padding: 5px;")
        self.show_signs_btn.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white;")
        self.quit_btn.setStyleSheet("padding: 5px; background-color: #f44336; color: white;")

        # Thread Setup
        self.thread = VideoThread(MODEL_PATH, FONT_PATHS, TRANSLATION_MAP)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_signs_signal.connect(self.update_detected_signs)
        self.thread.start()

        self.detected_signs = []
        self.current_lang = 'en'

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled)

    def change_language(self):
        lang_map = {0: 'en', 1: 'hi', 2: 'mr', 3: 'pa', 4: 'ta'}
        self.current_lang = lang_map[self.lang_combo.currentIndex()]
        self.thread.set_language(self.current_lang)

    def update_detected_signs(self, signs):
        self.detected_signs = signs

    def show_detected_signs(self):
        dialog = SignsDialog(self.detected_signs, self.current_lang, self)
        dialog.exec_()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    '''

# -*- coding: utf-8 -*-
import sys
import cv2
import time
import numpy as np
from collections import Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QComboBox, 
                             QPushButton, QVBoxLayout, QWidget, QDialog)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import google.generativeai as genai

# Constants
FPS = 2
MAX_SIGNS = 5
ANALYSIS_DURATION = 1
MODEL_PATH = 'Kartik_model_file.pt'
GEMINI_API_KEY = "AIzaSyDwqyUxpo7sy-1y1SgfXQtc2FwEIR9VdP8"

# Font Paths for Different Languages
FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"
}

# Updated Translation Map
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

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

class GeminiThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            response = gemini_model.generate_content(self.prompt)
            if response.text:
                self.finished.emit(response.text)
            else:
                self.error.emit("Empty response from API")
        except Exception as e:
            self.error.emit(f"Translation Error: {str(e)}")

class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray)
    update_signs = pyqtSignal(list)

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

    def run(self):
        while self.running:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame)
            annotated_frame = self.draw_annotations(frame, results)
            self.change_pixmap.emit(annotated_frame)

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
                    self.update_signs.emit(self.detected_signs.copy())
            return [], current_time
        return self.gesture_buffer, self.start_time

    def clear_history(self):
        self.detected_signs = []
        self.gesture_buffer = []
        self.update_signs.emit([])

    def set_language(self, lang_code):
        self.lang_code = lang_code
        self.clear_history()

    def stop(self):
        self.running = False
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Detection")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video Display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        

        # Language Selection
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Hindi", "Marathi", "Punjabi", "Tamil"])
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        self.layout.addWidget(self.lang_combo)
         # Add status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #333;")
        self.layout.addWidget(self.status_label)

        # Buttons
        self.translate_btn = QPushButton("Translate Sentence")
        self.translate_btn.clicked.connect(self.translate_sentence)
        self.layout.addWidget(self.translate_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        self.layout.addWidget(self.quit_btn)

        # Styling
        self.setStyleSheet("""
            background-color: #f0f0f0;
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QComboBox { padding: 5px; }
        """)
        self.translate_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.quit_btn.setStyleSheet("background-color: #f44336; color: white;")

        # Thread Setup
        self.thread = VideoThread(MODEL_PATH, FONT_PATHS, TRANSLATION_MAP)
        self.thread.change_pixmap.connect(self.update_image)
        self.thread.update_signs.connect(self.update_detected_signs)
        self.thread.start()

        self.detected_signs = []
        self.current_lang = 'en'
        self.is_translating = False

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def change_language(self):
        lang_map = {
            0: 'en', 1: 'hi', 2: 'mr', 
            3: 'pa', 4: 'ta'
        }
        self.current_lang = lang_map[self.lang_combo.currentIndex()]
        self.thread.set_language(self.current_lang)
        self.detected_signs = []
        self.update_display()

    def update_detected_signs(self, signs):
        if not self.is_translating:
            self.detected_signs = signs.copy()
            self.update_display()

    def update_display(self):
        if self.detected_signs:
            translated = [
                TRANSLATION_MAP[self.current_lang].get(s, s)
                for s in self.detected_signs
            ]
            self.video_label.setToolTip("\n".join(translated))
        else:
            self.video_label.setToolTip("Waiting for signs...")

    def translate_sentence(self):
        if not self.detected_signs:
            self.show_status_message("No signs detected!", "red", 2000)
            return
        
        self.is_translating = True
        self.translate_btn.setEnabled(False)
        sentence = ' '.join(self.detected_signs)
        lang_name = self.lang_combo.currentText()
        
        prompt =  f"""
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

      
        
        self.gemini_thread = GeminiThread(prompt)
        self.gemini_thread.finished.connect(self.handle_translation)
        self.gemini_thread.error.connect(self.handle_translation_error)
        self.gemini_thread.start()
        self.show_status_message("Processing translation...", "blue", 0)


    def handle_translation(self, result):
        self.show_status_message(result, "green", 5000)
        QTimer.singleShot(5000, self.reset_detection)
        self.translate_btn.setEnabled(True)
        self.is_translating = False


    def handle_translation_error(self, error_msg):
        self.show_status_message(error_msg, "red", 5000)
        self.translate_btn.setEnabled(True)
        self.is_translating = False

    def show_temp_message(self, message, duration):
        self.video_label.setToolTip(message)
        if duration > 0:
            QTimer.singleShot(duration, self.update_display)

    def show_status_message(self, message, color, duration):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"font-size: 16px; color: {color};")
        if duration > 0:
            QTimer.singleShot(duration, lambda: self.status_label.setText("Ready"))

    def reset_detection(self):
        self.thread.clear_history()
        self.detected_signs = []
        self.status_label.setText("Ready")

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())