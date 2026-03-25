from pathlib import Path
from ultralytics import YOLO
from config import (
    DATASET_PATH, YOLO_MODEL_NAME, TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE, TRAIN_IMG_SIZE, TRAIN_MODEL_NAME
)
from logger import log_info, log_error, log_warning

def train_model():
    """
    Huấn luyện YOLO model để phát hiện biển số xe
    """
    try:
        print("=" * 50)
        print("🚀 YOLO License Plate Training")
        print("=" * 50)
        
        # Kiểm tra dataset
        log_info(f"📂 Kiểm tra dataset: {DATASET_PATH}")
        if not Path(DATASET_PATH).exists():
            log_error(f"❌ Không tìm thấy dataset: {DATASET_PATH}")
            print("\n⚠️ Vui lòng chuẩn bị dataset tại: dataset/data.yaml")
            print("📖 Hướng dẫn chuẩn bị dataset:")
            print("   1. Tạo thư mục: dataset/images/train, dataset/images/val")
            print("   2. Tạo file data.yaml với nội dung:")
            print("""
path: dataset
train: images/train
val: images/val
nc: 1
names: ['plate']
            """)
            return False
        
        log_info("✅ Dataset tìm thấy")
        
        # Tải model
        log_info(f"🤖 Đang tải model: {YOLO_MODEL_NAME}")
        model = YOLO(YOLO_MODEL_NAME)
        log_info("✅ Model tải thành công")
        
        # Cấu hình huấn luyện
        print("\n" + "=" * 50)
        print("⚙️  Cấu hình Hu   n luyện")
        print("=" * 50)
        print(f"📊 Epochs: {TRAIN_EPOCHS}")
        print(f"📦 Batch Size: {TRAIN_BATCH_SIZE}")
        print(f"📐 Image Size: {TRAIN_IMG_SIZE}")
        print(f"💾 Model Name: {TRAIN_MODEL_NAME}")
        print("=" * 50)
        
        # Bắt đầu huấn luyện
        log_info("🔄 Bắt đầu huấn luyện...")
        print("\n💡 Quá trình huấn luyện đang chạy...\n")
        
        results = model.train(
            data=str(DATASET_PATH),
            epochs=TRAIN_EPOCHS,
            imgsz=TRAIN_IMG_SIZE,
            batch=TRAIN_BATCH_SIZE,
            name=TRAIN_MODEL_NAME,
            patience=10,  # Early stopping
            device=0,  # GPU device (0 = GPU đầu tiên, -1 = CPU)
            verbose=True,
            save=True,
            augment=True,
        )
        
        log_info("✅ Huấn luyện hoàn tất!")
        print("\n" + "=" * 50)
        print("✅ Huấn luyện thành công!")
        print("=" * 50)
        print(f"📁 Model được lưu tại: runs/detect/{TRAIN_MODEL_NAME}")
        
        return True
    
    except FileNotFoundError as e:
        log_error("❌ Không tìm thấy file cần thiết", e)
        return False
    
    except Exception as e:
        log_error("❌ Lỗi trong quá trình huấn luyện", e)
        return False

def validate_model(model_path=None):
    """
    Kiểm tra độ chính xác của model
    """
    try:
        if model_path is None:
            model_path = f"runs/detect/{TRAIN_MODEL_NAME}/weights/best.pt"
        
        log_info(f"🔍 Kiểm tra model: {model_path}")
        
        if not Path(model_path).exists():
            log_error(f"❌ Không tìm thấy model: {model_path}")
            return False
        
        model = YOLO(model_path)
        
        # Validate
        metrics = model.val(data=str(DATASET_PATH))
        
        log_info("✅ Kiểm tra hoàn tất")
        return True
    
    except Exception as e:
        log_error("Lỗi khi kiểm tra model", e)
        return False

def main():
    """Main function"""
    print("\n📋 Chọn hành động:")
    print("1. Huấn luyện model mới")
    print("2. Kiểm tra model")
    print("3. Thoát")
    
    choice = input("\n👉 Nhập lựa chọn (1-3): ").strip()
    
    if choice == "1":
        train_model()
    elif choice == "2":
        validate_model()
    elif choice == "3":
        log_info("👋 Thoát chương trình")
    else:
        print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()