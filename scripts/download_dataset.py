import os
import argparse
from roboflow import Roboflow
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="📥 Tải Dataset từ Roboflow")
    parser.add_argument('--workspace', type=str, default='springwangs-workspace', help='Tên Roboflow Workspace')
    parser.add_argument('--project', type=str, required=True, help='Tên Roboflow Project')
    parser.add_argument('--version', type=int, required=True, help='Phiên bản Dataset (VD: 1, 2, 3)')
    parser.add_argument('--target_folder', type=str, default='01_positive_standard', 
                        choices=['01_positive_standard', '02_Alley_Context', '03_Negative_Hard_Samples', '04_SAHI_Small_Objects'],
                        help='Thư mục đích trong data/')

    args = parser.parse_args()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print("❌ LỖI: Không tìm thấy ROBOFLOW_API_KEY trong file .env")
        return

    print(f"🚀 Đang kết nối tới Roboflow: {args.project} (v{args.version})...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8")

    # Đường dẫn đích
    target_path = os.path.join("data", args.target_folder)
    os.makedirs(target_path, exist_ok=True)

    print(f"📦 Đang di chuyển dữ liệu vào: {target_path}...")
    
    # Di chuyển nội dung từ thư mục vừa tải về (mặc định Roboflow tải về thư mục cùng tên project)
    downloaded_path = dataset.location
    
    # Ở đây chúng ta sẽ gom tất cả (train/val/test) vào folder đích 
    # vì hệ thống trainer.py của chúng ta sẽ tự động chia lại theo tỷ lệ config.
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(downloaded_path, split)
        if not os.path.exists(split_dir): continue
        
        # Copy images
        img_src = os.path.join(split_dir, 'images')
        img_dst = os.path.join(target_path, 'images')
        os.makedirs(img_dst, exist_ok=True)
        if os.path.exists(img_src):
            os.system(f"cp -r {img_src}/* {img_dst}/")
            
        # Copy labels
        label_src = os.path.join(split_dir, 'labels')
        label_dst = os.path.join(target_path, 'labels')
        os.makedirs(label_dst, exist_ok=True)
        if os.path.exists(label_src):
            os.system(f"cp -r {label_src}/* {label_dst}/")

    print(f"✅ Tải và chuẩn bị dữ liệu thành công cho {args.target_folder}!")

if __name__ == "__main__":
    main()
