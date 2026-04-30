import os
import argparse
import subprocess
from pathlib import Path
import yaml

def load_raw_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_raw_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="🚀 A/B Test Runner cho Fire Detection")
    parser.add_argument('--models', nargs='+', choices=['baseline', 'data_plus_plus', 'sota'], 
                        default=['baseline', 'data_plus_plus', 'sota'],
                        help='Danh sách các model muốn test')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Đường dẫn file config')
    parser.add_argument('--stage', type=str, default='all', choices=['baseline', 'hard_negative', 'sahi', 'all'],
                        help='Stage huấn luyện muốn chạy cho mỗi model')
    
    args = parser.parse_args()
    original_config_path = Path(args.config)
    
    print(f"\n{'#'*60}")
    print(f"🔥 BẮT ĐẦU QUÁ TRÌNH A/B TESTING")
    print(f"   Models to test: {args.models}")
    print(f"   Stage: {args.stage}")
    print(f"{'#'*60}\n")

    for model_name in args.models:
        print(f"\n🚀 Đang huấn luyện Model: {model_name.upper()}")
        
        # 1. Tạm thời sửa config để active model này
        config_data = load_raw_config(original_config_path)
        config_data['model']['active_variant'] = model_name
        
        temp_config_path = original_config_path.parent / f"temp_{model_name}.yaml"
        save_raw_config(config_data, temp_config_path)
        
        # 2. Gọi trainer script
        cmd = [
            "python", "-m", "src.engine.trainer",
            "--stage", args.stage,
            "--config", str(temp_config_path)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Đã huấn luyện xong Model: {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi huấn luyện {model_name}: {e}")
        finally:
            # Dọn dẹp config tạm
            if temp_config_path.exists():
                os.remove(temp_config_path)

    print(f"\n{'#'*60}")
    print(f"🎉 TẤT CẢ CÁC MODEL ĐÃ ĐƯỢC HUẤN LUYỆN XONG!")
    print(f"   Bạn có thể xem kết quả so sánh trong thư mục runs/")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
