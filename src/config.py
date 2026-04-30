"""
============================================================
📋 Config Loader — Đọc và quản lý cấu hình hệ thống
============================================================

MỤC ĐÍCH:
    Đọc file YAML config và chuyển thành object Python.
    Hỗ trợ truy cập bằng dot-notation (vd: config.model.img_size)
    thay vì dict notation (config['model']['img_size']).

GIẢI THÍCH CHO BẠN:
    1. Config class kế thừa từ dict → vừa là dict, vừa truy cập được bằng dấu chấm
    2. load_config() đọc file YAML → trả về Config object
    3. Merge với environment variables từ .env (cho API keys)

VÍ DỤ SỬ DỤNG:
    >>> config = load_config('configs/default.yaml')
    >>> print(config.model.architecture)   # 'rtdetr-l'
    >>> print(config.training.epochs)      # 100
    >>> print(config.sahi.slice_height)    # 320
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


class Config(dict):
    """
    Config class — cho phép truy cập cấu hình bằng dot-notation.

    KẾ THỪA TỪ dict:
        Vì Python dict chỉ hỗ trợ config['key'], nhưng mình muốn
        gõ config.key cho tiện. Class này override __getattr__ và
        __setattr__ để làm điều đó.

    VÍ DỤ:
        >>> c = Config({'model': {'img_size': 640}})
        >>> c.model.img_size  # 640 — dùng dấu chấm
        >>> c['model']['img_size']  # 640 — vẫn dùng dict được
    """

    def __init__(self, *args, **kwargs):
        """
        Khởi tạo Config object.
        Đệ quy chuyển tất cả dict lồng nhau thành Config objects.
        """
        super(Config, self).__init__(*args, **kwargs)
        # Duyệt qua tất cả key-value
        for key, value in self.items():
            if isinstance(value, dict):
                # Nếu value là dict → chuyển thành Config (đệ quy)
                self[key] = Config(value)
            elif isinstance(value, list):
                # Nếu value là list → kiểm tra từng phần tử
                self[key] = [
                    Config(item) if isinstance(item, dict) else item
                    for item in value
                ]

    def __getattr__(self, key):
        """
        Cho phép truy cập bằng dấu chấm: config.model
        Khi Python không tìm thấy attribute thông thường,
        nó sẽ gọi __getattr__ → ta trả về self[key].
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"Config không có thuộc tính '{key}'. "
                f"Các key có sẵn: {list(self.keys())}"
            )

    def __setattr__(self, key, value):
        """Cho phép gán bằng dấu chấm: config.model = ..."""
        self[key] = value

    def __delattr__(self, key):
        """Cho phép xóa bằng dấu chấm: del config.model"""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config không có thuộc tính '{key}'")

    def to_dict(self):
        """
        Chuyển Config về dict thuần túy (để serialize hay save).
        Đệ quy chuyển tất cả Config con thành dict.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Đọc file YAML config và trả về Config object.

    FLOW:
        1. Load biến môi trường từ .env (nếu có)
        2. Đọc file YAML
        3. Override config bằng environment variables (cho API keys)
        4. Trả về Config object

    Args:
        config_path: Đường dẫn tới file YAML config

    Returns:
        Config object với dot-notation access

    Raises:
        FileNotFoundError: Nếu file config không tồn tại
    """
    # --- Bước 1: Load .env file ---
    # .env chứa API keys nhạy cảm, không push lên Git
    load_dotenv()

    # --- Bước 2: Đọc file YAML ---
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file config: {config_path}\n"
            f"Hãy chắc chắn file configs/default.yaml tồn tại."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # --- Bước 3: Tạo Config object ---
    config = Config(config_dict)

    # --- Bước 4: Override bằng environment variables ---
    # API keys lấy từ .env file, override vào config
    _override_from_env(config)

    return config


def _override_from_env(config: Config):
    """
    Override config bằng biến môi trường.

    LÝ DO:
        API keys không nên hardcode trong YAML config.
        Thay vào đó, đặt trong .env file (không push lên Git)
        và override ở đây.

    Args:
        config: Config object (sẽ bị modify in-place)
    """
    # --- Telegram ---
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        config.alert.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        config.alert.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- Twilio ---
    if os.getenv("TWILIO_ACCOUNT_SID"):
        config.alert.twilio.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        config.alert.twilio.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        config.alert.twilio.from_number = os.getenv("TWILIO_FROM_NUMBER")
        to_numbers = os.getenv("TWILIO_TO_NUMBERS", "")
        config.alert.twilio.to_numbers = [
            n.strip() for n in to_numbers.split(",") if n.strip()
        ]

    # --- Zalo OA ---
    if os.getenv("ZALO_ACCESS_TOKEN"):
        config.alert.zalo.access_token = os.getenv("ZALO_ACCESS_TOKEN")
        config.alert.zalo.user_id = os.getenv("ZALO_USER_ID", "")

    # --- Vietmap ---
    if os.getenv("VIETMAP_API_KEY"):
        config.alert.vietmap.api_key = os.getenv("VIETMAP_API_KEY")

    # --- WandB ---
    if os.getenv("WANDB_API_KEY"):
        config.wandb.api_key = os.getenv("WANDB_API_KEY")


def get_device(config: Config) -> str:
    """
    Xác định device để chạy model (GPU, CPU, hay Apple MPS).

    LOGIC:
        - "auto": Tự động chọn → CUDA > MPS > CPU
        - "cuda": Dùng NVIDIA GPU
        - "mps": Dùng Apple Silicon GPU (M1/M2/M3)
        - "cpu": Dùng CPU (chậm nhất)

    Args:
        config: Config object chứa inference.device

    Returns:
        Device string: "cuda", "mps", hoặc "cpu"
    """
    import torch

    device_setting = config.inference.device

    if device_setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon (M1/M2/M3)
        else:
            return "cpu"
    else:
        return device_setting
