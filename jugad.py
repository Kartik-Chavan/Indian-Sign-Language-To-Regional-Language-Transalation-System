# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtCore import Qt

FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"
}

COLORS = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A133FF"]

class ColorCycler:
    def __init__(self, colors):
        self.colors = colors
        self.index = 0

    def next(self):
        color = self.colors[self.index]
        self.index = (self.index + 1) % len(self.colors)
        return color

class SentenceDialog(QDialog):
    def __init__(self, sentences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Translated Sentences")
        self.setStyleSheet("background-color: #f0f0f0; padding: 20px;")

        layout = QVBoxLayout()
        color_cycler = ColorCycler(COLORS)

        for lang_code, sentence in sentences.items():
            label = QLabel(sentence)
            font_id = QFontDatabase.addApplicationFont(FONT_PATHS[lang_code])
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            label.setFont(QFont(font_family, 20))
            label.setStyleSheet(f"color: {color_cycler.next()}; margin-bottom: 10px;")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

        self.setLayout(layout)

    def closeEvent(self, event):
        """Ensure the dialog destroys itself and exits cleanly."""
        self.destroy()
        QApplication.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    sentences = {
        'ta': "நான் இருமலுக்கு மருந்து கோரிக்கை செய்கிறேன்.",
      
    }
   
    dialog = SentenceDialog(sentences)
    dialog.resize(400, 50)
    dialog.show()

    sys.exit(app.exec_())
