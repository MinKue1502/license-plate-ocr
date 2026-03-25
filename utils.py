import re
import cv2
import numpy as np
from config import CHAR_REPLACE_MAP, PLATE_PATTERN
from logger import log_error, log_warning

def clean_text(text):
    """
    Làm sạch text từ OCR
    - Chuyển thành chữ hoa
    - Thay thế các ký tự nhầm lẫn
    - Xóa ký tự không hợp lệ
    """
    try:
        if not text:
            return ""
        
        text = text.upper().strip()
        
        # Thay thế ký tự nhầm lẫn
        for old_char, new_char in CHAR_REPLACE_MAP.items():
            text = text.replace(old_char, new_char)
        
        # Xóa ký tự không hợp lệ (chỉ giữ lại chữ, số, dấu chấm, dấu gạch)
        text = re.sub(r"[^A-Z0-9.-]", "", text)
        
        return text.strip()
    
    except Exception as e:
        log_error("Lỗi khi làm sạch text", e)
        return ""

def validate_plate(text):
    """
    Kiểm tra định dạng biển số Việt Nam
    Format: 29A-12345 hoặc 51F-888.88
    """
    try:
        if not text:
            return []
        
        matches = re.findall(PLATE_PATTERN, text)
        return matches
    
    except Exception as e:
        log_error("Lỗi khi kiểm tra định dạng biển số", e)
        return []

def preprocess_image(image, resize_scale=2):
    """
    Tiền xử lý ảnh để cải thiện OCR
    - Chuyển sang grayscale
    - Cân bằng histogram
    - Tăng độ sắc nét
    - Nhị phân hóa
    - Phóng to ảnh
    """
    try:
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cân bằng histogram
        gray = cv2.equalizeHist(gray)
        
        # Tăng độ sắc nét với kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        
        # Nhị phân hóa với OTSU
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Phóng to ảnh để OCR dễ đọc hơn
        if resize_scale > 1:
            h, w = thresh.shape
            thresh = cv2.resize(thresh, (w * resize_scale, h * resize_scale))
        
        return thresh
    
    except Exception as e:
        log_error("Lỗi khi tiền xử lý ảnh", e)
        return None

def crop_and_pad(image, box, pad_ratio=0.2):
    """
    Cắt vùng biển số từ ảnh gốc và thêm padding
    
    Args:
        image: ảnh gốc
        box: bounding box từ YOLO (x1, y1, x2, y2)
        pad_ratio: tỷ lệ padding (0.2 = 20%)
    
    Returns:
        Ảnh đã cắt và thêm padding
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # Tính padding
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        
        # Áp dụng padding với giới hạn
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Cắt ảnh
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    except Exception as e:
        log_error("Lỗi khi cắt ảnh", e)
        return None

def draw_detection_box(image, box, label="Plate", color=(0, 255, 0), thickness=2):
    """
    Vẽ bounding box trên ảnh
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # Vẽ box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Vẽ label
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1, y1 - 10), font, 0.9, color, 2)
        
        return image
    
    except Exception as e:
        log_error("Lỗi khi vẽ bounding box", e)
        return image

def is_valid_plate_format(text):
    """
    Kiểm tra xem text có phải là biển số hợp lệ không
    """
    try:
        if not text or len(text) < 3:
            return False
        
        matches = validate_plate(text)
        return len(matches) > 0
    
    except Exception as e:
        log_error("Lỗi khi kiểm tra định dạng biển số", e)
        return False