import os
from roboflow import Roboflow

def main():
    print("="*50)
    print("🚀 MLOps Pipeline: Trích xuất và Tải Dataset về Đóng gói")
    print("="*50)
    
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace_name = os.environ.get("ROBOFLOW_WORKSPACE")
    project_name = os.environ.get("ROBOFLOW_PROJECT")
    version_idx = int(os.environ.get("VERSION_IDX", "1"))
    
    if not api_key:
        print("LỖI: Rớt kết nối do thiếu ROBOFLOW_API_KEY trong Github Secrets.")
        return

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_name).project(project_name)
        
        print(f"🔄 Đang kết nối vào bản Version {version_idx} trên Cloud...")
        version = project.version(version_idx)
        
        print("📦 Đang Download toàn bộ hình ảnh và nhãn thành chuẩn YOLOv8 để đóng gói ZIP...")
        # Lệnh này tải dữ liệu về máy để Github tiến hành nén lại
        dataset = version.download("yolov8", location=f"dataset_export_v{version_idx}")
        print(f"✅ Tải thành công! Dữ liệu đã sẵn sàng để đưa lên Google Drive.")
        
    except Exception as e:
        print(f"❌ Thu thập dữ liệu thất bại: {e}")

if __name__ == "__main__":
    main()
