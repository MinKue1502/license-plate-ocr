import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from pathlib import Path
from config import (
    PAGE_TITLE, PAGE_LAYOUT, TESSERACT_PATH, MODEL_PATH,
    YOLO_CONFIDENCE, OCR_CONFIDENCE, OCR_LANGUAGES
)
from logger import log_info, log_error, log_warning
from utils import (
    crop_and_pad, preprocess_image, clean_text, validate_plate,
    is_valid_plate_format, draw_detection_box
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log_error("⚠️ YOLO không được cài đặt")

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)
st.title(PAGE_TITLE)

# ===== SETUP TESSERACT =====
if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    log_info(f"✅ Tesseract tìm thấy: {TESSERACT_PATH}")
else:
    log_warning("⚠️ Tesseract không được cài đặt hoặc không tìm thấy")
    st.warning("⚠️ Tesseract không được cài đặt. Vui lòng cài đặt để sử dụng OCR Pytesseract.")

# ===== LOAD MODELS =====
@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    try:
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO không được cài đặt")
            return None
        
        if not Path(MODEL_PATH).exists():
            st.error(f"❌ Model không tìm thấy: {MODEL_PATH}")
            return None
        
        log_info(f"🤖 Đang tải YOLO model: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        log_info("✅ YOLO model tải thành công")
        return model
    except Exception as e:
        log_error("Lỗi khi tải YOLO model", e)
        st.error(f"❌ Lỗi khi tải YOLO model: {str(e)}")
        return None

@st.cache_resource
def load_easyocr_reader():
    """Load EasyOCR reader"""
    try:
        log_info("🤖 Đang tải EasyOCR reader...")
        reader = easyocr.Reader(OCR_LANGUAGES)
        log_info("✅ EasyOCR reader tải thành công")
        return reader
    except Exception as e:
        log_error("Lỗi khi tải EasyOCR reader", e)
        st.error(f"❌ Lỗi khi tải EasyOCR reader: {str(e)}")
        return None

# Load models
yolo_model = load_yolo_model()
ocr_reader = load_easyocr_reader()

# ===== OCR FUNCTION =====
def perform_ocr(image):
    """Thực hiện OCR với dual engine"""
    results = {
        "easyocr": "",
        "tesseract": "",
        "best": ""
    }
    
    try:
        # EasyOCR
        if ocr_reader:
            try:
                log_info("📖 Đang sử dụng EasyOCR...")
                ocr_result = ocr_reader.readtext(image)
                if ocr_result:
                    results["easyocr"] = ocr_result[0][1]
            except Exception as e:
                log_warning(f"EasyOCR lỗi: {str(e)}")
        
        # Tesseract (fallback)
        if TESSERACT_PATH and Path(TESSERACT_PATH).exists():
            try:
                log_info("📖 Đang sử dụng Tesseract...")
                configs = ["--psm 7", "--psm 8", "--psm 6"]
                best_text = ""
                for cfg in configs:
                    t = pytesseract.image_to_string(
                        image,
                        config=f"{cfg} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
                    ).strip()
                    if len(t) > len(best_text):
                        best_text = t
                results["tesseract"] = best_text
            except Exception as e:
                log_warning(f"Tesseract lỗi: {str(e)}")
        
        # Chọn kết quả tốt nhất
        if len(results["easyocr"]) >= len(results["tesseract"]):
            results["best"] = results["easyocr"]
        else:
            results["best"] = results["tesseract"]
        
    except Exception as e:
        log_error("Lỗi trong quá trình OCR", e)
    
    return results

# ===== MAIN APP =====
st.markdown("---")

# Upload section
col_upload = st.columns(1)[0]
with col_upload:
    uploaded_file = st.file_uploader("📤 Tải lên ảnh biển số xe", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file:
    try:
        log_info(f"📤 File tải lên: {uploaded_file.name}")
        
        # Load image
        image = Image.open(uploaded_file)
        img = np.array(image)
        
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        h, w = img.shape[:2]
        log_info(f"✅ Ảnh tải thành công. Size: {w}x{h}")
        
        # YOLO Detection
        if yolo_model is None:
            st.error("❌ YOLO model không khả dụng. Vui lòng kiểm tra cấu hình.")
        else:
            log_info("🔍 Đang phát hiện biển số...")
            
            col1, col2 = st.columns(2)
            
            results = yolo_model(img, conf=YOLO_CONFIDENCE)
            img_draw = img.copy()
            plates_data = []
            
            for r in results:
                if len(r.boxes) == 0:
                    log_warning("⚠️ Không phát hiện biển số nào")
                    st.warning("⚠️ Không phát hiện biển số nào")
                    continue
                
                for idx, box in enumerate(r.boxes.xyxy):
                    try:
                        # Cắt ảnh
                        crop = crop_and_pad(img, box, pad_ratio=0.2)
                        if crop is None:
                            continue
                        
                        # Tiền xử lý
                        thresh = preprocess_image(crop, resize_scale=2)
                        if thresh is None:
                            continue
                        
                        # OCR
                        ocr_results = perform_ocr(thresh)
                        text = clean_text(ocr_results["best"])
                        
                        plates_data.append({
                            "index": idx + 1,
                            "box": box,
                            "crop": crop,
                            "thresh": thresh,
                            "raw_text": ocr_results["best"],
                            "clean_text": text,
                            "easyocr": ocr_results["easyocr"],
                            "tesseract": ocr_results["tesseract"]
                        })
                        
                        # Vẽ box
                        img_draw = draw_detection_box(img_draw, box, f"Plate {idx+1}", (0, 255, 0), 2)
                        
                    except Exception as e:
                        log_error(f"Lỗi xử lý biển số {idx+1}", e)
                        continue
            
            # Display results
            with col1:
                st.subheader("📷 Ảnh Gốc + Box")
                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col2:
                st.subheader("📊 Kết Quả OCR")
                
                if len(plates_data) == 0:
                    st.error("❌ Không nhận diện được biển số")
                else:
                    for plate in plates_data:
                        st.write(f"**Biển số {plate['index']}:**")
                        
                        # Hiển thị ảnh cắt
                        st.image(plate["crop"], caption=f"Crop {plate['index']}", width=200)
                        st.image(plate["thresh"], caption=f"Sau xử lý {plate['index']}", width=200)
                        
                        # Hiển thị text
                        st.write(f"🔤 Raw: `{plate['raw_text']}`")
                        st.write(f"🧹 Clean: `{plate['clean_text']}`")
                        
                        # Kiểm tra định dạng
                        if is_valid_plate_format(plate['clean_text']):
                            valid = validate_plate(plate['clean_text'])
                            st.success(f"✅ Hợp lệ: {valid[0]}")
                        else:
                            st.warning("⚠️ Sai định dạng")
                        
                        st.divider()
            
            # Final result
            if len(plates_data) > 0:
                st.markdown("---")
                st.subheader("🚗 Tổng Hợp Biển Số")
                
                final_plates = [p["clean_text"] for p in plates_data if is_valid_plate_format(p["clean_text"])]
                
                if final_plates:
                    for i, plate in enumerate(final_plates, 1):
                        st.success(f"**Biển số {i}:** `{plate}`")
                else:
                    st.info("ℹ️ Không có biển số hợp lệ")
            
            log_info(f"✅ Phát hiện xong. Tìm thấy {len(plates_data)} biển số")
    
    except Exception as e:
        log_error("Lỗi chung trong ứng dụng", e)
        st.error(f"❌ Lỗi: {str(e)}")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
### 📖 Hướng dẫn sử dụng:
1. Tải lên ảnh biển số xe (JPG, PNG, JPEG, BMP)
2. Hệ thống sẽ:
   - 🔍 Phát hiện biển số bằng YOLO
   - 🧹 Tiền xử lý ảnh (grayscale, histogram, sharpen, threshold)
   - 📖 Nhận diện text bằng EasyOCR + Tesseract
   - ✅ Kiểm tra định dạng biển số Việt Nam
3. Xem kết quả ở bên phải

### 🎯 Format biển số Việt Nam:
- **Cũ:** `29A-12345` (2 số + 1 chữ + 5 số)
- **Mới:** `51F-888.88` (2 số + 1 chữ + 3-5 số + 2 số thập phân)

### ⚙️ Cài đặt cần thiết:
- **Tesseract:** Cài đặt từ https://github.com/UB-Mannheim/tesseract/wiki
- **Python:** >= 3.8
- **GPU:** Tối ưu nhất nhưng không bắt buộc

---
*Phát triển bởi: License Plate OCR Team* 🚀
""")