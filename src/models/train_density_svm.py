import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("        HUẤN LUYỆN MÔ HÌNH SVM PHÂN LOẠI MẬT ĐỘ GIAO THÔNG DYNAMIC        ")
    print("=" * 60)
    
    # 1. Đọc dữ liệu đặc trưng đã trích xuất
    features_csv = "data/traffic_light/density_features.csv"
    
    if not Path(features_csv).exists():
        print(f"[Lỗi] Không tìm thấy file {features_csv}. Vui lòng chạy collect_density_features.py trước.")
        return
        
    df = pd.read_csv(features_csv)
    print(f"Đã tải {len(df)} mẫu từ file dữ liệu đặc trưng {features_csv}.")
    
    # Tách X (features) và y (labels)
    # Các đặc trưng: motorcycle_count, car_count, stopped_vehicles, pcu_load, average_speed
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Chia tập huấn luyện và đánh giá (80% train, 20% test)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Số lượng mẫu huấn luyện (Train): {len(X_train)}")
    print(f"Số lượng mẫu đánh giá (Val): {len(X_val)}")
    
    # 2. Chuẩn hóa đặc trưng (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Khởi tạo và huấn luyện mô hình SVM
    # Sử dụng kernel RBF, bật probability=True để có thể dự đoán xác suất tin cậy nếu cần
    print("[Huấn luyện] Đang huấn luyện mô hình SVM mật độ với nhân RBF...")
    svm = SVC(kernel='rbf', C=2.0, class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    print("[Hoàn tất] Huấn luyện hoàn thành!")
    
    # 4. Dự đoán và đánh giá
    y_pred = svm.predict(X_val_scaled)
    
    accuracy = accuracy_score(y_val, y_pred)
    print("\n" + "-" * 50)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ (ĐỘ CHÍNH XÁC ACCURACY): {accuracy:.4f}")
    print("-" * 50)
    
    class_names = ['Low/Empty', 'Medium/Normal', 'High/Congested']
    unique_labels = sorted(list(set(y_val).union(set(y_pred))))
    target_names = [class_names[i] for i in unique_labels]
    
    print("\nBÁO CÁO CHI TIẾT HIỆU NĂNG PHÂN LOẠI (CLASSIFICATION REPORT):")
    print(classification_report(y_val, y_pred, labels=unique_labels, target_names=target_names))
    
    print("\nMA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    conf_mat = confusion_matrix(y_val, y_pred, labels=unique_labels)
    print(f"{'Thực tế / Dự đoán':<20}", end="")
    for name in target_names:
        print(f"{name:>15}", end="")
    print()
    for i, row in enumerate(conf_mat):
        print(f"{target_names[i]:<20}", end="")
        for val in row:
            print(f"{val:>15}", end="")
        print()
    print("-" * 50)
    
    # 5. Lưu mô hình và bộ chuẩn hóa (pickle)
    model_save_dir = Path("src/models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / "svm_traffic_density.pkl"
    
    save_data = {
        'scaler': scaler,
        'svm': svm,
        'classes': class_names
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"\n[Thành công] Đã lưu mô hình SVM phân loại mật độ tại: {model_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
