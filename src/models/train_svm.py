import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("        HUẤN LUYỆN THUẬT TOÁN HỌC MÁY SVM (SUPPORT VECTOR MACHINE)        ")
    print("=" * 60)
    
    # 1. Đọc dữ liệu đặc trưng đã trích xuất
    train_path = "data/traffic_light/train_features.csv"
    val_path = "data/traffic_light/val_features.csv"
    
    if not Path(train_path).exists() or not Path(val_path).exists():
        print("[Lỗi] Không tìm thấy file dữ liệu đặc trưng CSV. Vui lòng chạy extract_features.py trước.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Đã tải {len(train_df)} mẫu huấn luyện và {len(val_df)} mẫu đánh giá.")
    
    # Tách X (features) và y (labels)
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    
    X_val = val_df.drop(columns=['label']).values
    y_val = val_df['label'].values
    
    # 2. Chuẩn hóa đặc trưng (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Khởi tạo và huấn luyện mô hình SVM
    # Sử dụng kernel RBF phi tuyến tính, bật probability=True để lấy xác suất tin cậy
    print("[Huấn luyện] Đang huấn luyện mô hình SVM với nhân RBF...")
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    print("[Hoàn tất] Huấn luyện hoàn thành!")
    
    # 4. Dự đoán và đánh giá
    y_pred = svm.predict(X_val_scaled)
    
    accuracy = accuracy_score(y_val, y_pred)
    print("\n" + "-" * 50)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ (ĐỘ CHÍNH XÁC ACCURACY): {accuracy:.4f}")
    print("-" * 50)
    
    class_names = ['green', 'off', 'red', 'yellow']
    # Lấy các nhãn thực tế xuất hiện trong tập val để tránh lỗi thiếu class trong classification_report
    unique_labels = sorted(list(set(y_val).union(set(y_pred))))
    target_names = [class_names[i] for i in unique_labels]
    
    print("\nBÁO CÁO CHI TIẾT HIỆU NĂNG PHÂN LOẠI (CLASSIFICATION REPORT):")
    print(classification_report(y_val, y_pred, labels=unique_labels, target_names=target_names))
    
    print("\nMA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    conf_mat = confusion_matrix(y_val, y_pred, labels=unique_labels)
    # In ma trận nhầm lẫn đẹp hơn
    print(f"{'Thực tế / Dự đoán':<20}", end="")
    for name in target_names:
        print(f"{name:>10}", end="")
    print()
    for i, row in enumerate(conf_mat):
        print(f"{target_names[i]:<20}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()
    print("-" * 50)
    
    # 5. Lưu mô hình và bộ chuẩn hóa (pickle)
    model_save_dir = Path("src/models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / "svm_traffic_light.pkl"
    
    save_data = {
        'scaler': scaler,
        'svm': svm,
        'classes': class_names
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"\n[Thành công] Đã lưu mô hình SVM & Scaler tại: {model_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
