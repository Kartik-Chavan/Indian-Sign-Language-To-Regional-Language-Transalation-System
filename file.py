import os
print(os.path.exists("./src/scripts/NotoSansGurmukhi-Regular.ttf"))


# Font Paths for Different Languages
FONT_PATHS = {
    'en': "./src/scripts/Arial Unicode MS.ttf",  # English
    'hi': "./src/scripts/NotoSansDevanagari-Regular.ttf",  # Hindi
    'mr': "./src/scripts/NotoSansDevanagari-Regular.ttf",  # Marathi
    'pa': "./src/scripts/NotoSansGurmukhi-Regular.ttf",  # Punjabi
    'ta': "./src/scripts/NotoSansTamil-Regular.ttf"  # Tamil
}

TRANSLATION_MAP = {
    'en': {'C': 'C', 'Friends': 'Friends', 'H': 'H', 'Hello': 'Hello', 'L': 'L', 'O': 'O', 'Please': 'Please', 'Thanks': 'Thanks'},
    'hi': {'C': 'सी', 'Friends': 'दोस्त', 'H': 'एच', 'Hello': 'नमस्ते', 'L': 'एल', 'O': 'ओ', 'Please': 'कृपया', 'Thanks': 'धन्यवाद'},
    'mr': {'C': 'सी', 'Friends': 'मित्र', 'H': 'एच', 'Hello': 'नमस्कार', 'L': 'एल', 'O': 'ओ', 'Please': 'कृपया', 'Thanks': 'धन्यवाद'},
    'pa': {'C': 'ਸੀ', 'Friends': 'ਦੋਸਤ', 'H': 'ਐਚ', 'Hello': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'L': 'ਐਲ', 'O': 'ਓ', 'Please': 'ਕ੍ਰਿਪਾ ਕਰਕੇ', 'Thanks': 'ਧੰਨਵਾਦ'},
    'ta': {'C': 'சி', 'Friends': 'நண்பர்கள்', 'H': 'எச்', 'Hello': 'வணக்கம்', 'L': 'எல்', 'O': 'ஓ', 'Please': 'தயவுசெய்து', 'Thanks': 'நன்றி'}
}

def 
