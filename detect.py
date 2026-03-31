import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
from config import MODEL_PATH, YOLO_CONFIDENCE
from logger import log_info, log_error
from utils import crop_and_pad, preprocess_image, clean_text, validate_plate

def detect_license_plates(image_path, model_path=None):
    """
    Phát hiện biển số xe từ ảnh
    
    Args:
        image_path: đường dẫn ảnh
        model_path: đường dẫn model YOLO (mặc định: config.MODEL_PATH)
    
    Returns:
        dict: kết quả phát hiện
    """
    
    try:
        # Xác định model path
        if model_path is None:
            model_path = MODEL_PATH
        
        # Kiểm tra ảnh tồn tại
        if not Path(image_path).exists():
            log_error(f" Không tìm thấy ảnh: {image_path}")
            return {"success": False, "error": "Ảnh không tồn tại"}
        
        log_info(f" Đang tải ảnh: {image_path}")
        img = cv2.imread(str(image_path))
        
        if img is None:
            log_error(f" Không thể đọc ảnh: {image_path}")
            return {"success": False, "error": "Không thể đọc ảnh"}
        
        h, w = img.shape[:2]
        log_info(f" Ảnh tải thành công. Size: {w}x{h}")
        
        # Kiểm tra model tồn tại
        if not Path(model_path).exists():
            log_error(f" Không tìm thấy model: {model_path}")
            return {"success": False, "error": "Model không tồn tại"}
        
        # Load YOLO model
        log_info(f" Đang tải model: {model_path}")
        model = YOLO(str(model_path))
        log_info(" Model tải thành công")
        
        # Phát hiện
        log_info(" Đang phát hiện biển số...")
        results = model(img, conf=YOLO_CONFIDENCE)
        
        plates_data = []
        img_draw = img.copy()
        
        for r in results:
            if len(r.boxes) == 0:
                log_info(" Không phát hiện biển số nào")
                continue
            
            for idx, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                
                # Vẽ box
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Cắt và xử lý
                crop = crop_and_pad(img, box, pad_ratio=0.2)
                if crop is None:
                    continue
                
                # Tiền xử lý
                thresh = preprocess_image(crop, resize_scale=2)
                if thresh is None:
                    continue
                
                plates_data.append({
                    "index": idx + 1,
                    "box": (x1, y1, x2, y2),
                    "crop": crop,
                    "preprocessed": thresh
                })
        
        log_info(f" Phát hiện xong. Tìm thấy {len(plates_data)} biển số")
        
        return {
            "success": True,
            "image": img,
            "image_draw": img_draw,
            "plates": plates_data,
            "count": len(plates_data)
        }
    
    except Exception as e:
        log_error("Lỗi trong quá trình phát hiện", e)
        return {"success": False, "error": str(e)}

def main():
    """Script test"""
    print("=" * 50)
    print(" License Plate Detection")
    print("=" * 50)
    
    # Nhập đường dẫn ảnh
    image_path = input("\n Nhập đường dẫn ảnh (ví dụ: test.jpg): ").strip()
    
    if not image_path:
        print(" Vui lòng nhập đường dẫn ảnh")
        return
    
    # Phát hiện
    result = detect_license_plates(image_path)
    
    if not result["success"]:
        print(f" Lỗi: {result['error']}")
        return
    
    # Hiển thị kết quả
    print(f"\n Phát hiện xong!")
    print(f" Tìm thấy {result['count']} biển số\n")
    
    # Hiển thị ảnh
    img_draw = result["image_draw"]
    cv2.imshow("Detection Result", img_draw)
    print(" Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()