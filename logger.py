import logging
import sys
from pathlib import Path
from datetime import datetime

# Tạo thư mục logs nếu chưa có
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Tạo tên file log với timestamp
log_file = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Lưu vào file
        logging.StreamHandler(sys.stdout)  # Hiển thị trên console
    ]
)

logger = logging.getLogger(__name__)

def log_info(message):
    """Log thông tin"""
    logger.info(message)

def log_error(message, error=None):
    """Log lỗi"""
    if error:
        logger.error(f"{message}: {str(error)}", exc_info=True)
    else:
        logger.error(message)

def log_warning(message):
    """Log cảnh báo"""
    logger.warning(message)

def log_debug(message):
    """Log debug"""
    logger.debug(message)