import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===== PATHS =====
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "data.yaml"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "plate_model" / "weights" / "best.pt"
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8s.pt")

# ===== TESSERACT =====
# Tự động phát hiện đường dẫn Tesseract
def get_tesseract_path():
    """Tìm đường dẫn Tesseract tự động"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",  # Linux
        "/usr/local/bin/tesseract",  # Mac
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Nếu không tìm thấy, return None (để người dùng cài đặt)
    return None

TESSERACT_PATH = os.getenv("TESSERACT_PATH") or get_tesseract_path()

# ===== OCR SETTINGS =====
OCR_CONFIDENCE = float(os.getenv("OCR_CONFIDENCE", 0.3))
OCR_LANGUAGES = ["en", "vi"]  # Support English and Vietnamese
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.3))

# ===== STREAMLIT SETTINGS =====
PAGE_TITLE = "🚗 Nhận diện Biển số Xe (YOLO + OCR)"
PAGE_LAYOUT = "wide"

# ===== TRAINING SETTINGS =====
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", 100))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 8))
TRAIN_IMG_SIZE = int(os.getenv("TRAIN_IMG_SIZE", 640))
TRAIN_MODEL_NAME = os.getenv("TRAIN_MODEL_NAME", "plate_model")

# ===== CHARACTER REPLACEMENT MAP =====
CHAR_REPLACE_MAP = {
    "G": "6", "I": "1", "B": "8", "O": "0", "S": "5",
    "i": "1", "o": "0", "s": "5", "b": "8", "g": "6"
}

# ===== PLATE VALIDATION PATTERN =====
# Định dạng biển số Việt Nam: 29A-12345 hoặc 51F-888.88
PLATE_PATTERN = r"\d{2}[A-Z]\d?-?\d{3,5}\.?\d{0,2}"