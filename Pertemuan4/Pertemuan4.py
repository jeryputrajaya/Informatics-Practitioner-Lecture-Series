"""
================================================================================
PERTEMUAN 4: Introduction to Machine Learning
Pemateri   : Taniza Tajudin (Universiti Teknologi MARA, Malaysia)
Tugas      : Implementasi Supervised vs Unsupervised Machine Learning
Program Studi Informatika - Universitas Bengkulu
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

def generate_student_ml_data():
    """
    PERTEMUAN 4 - DATA UNDERSTANDING:
    Membuat Dataset Simulasi Pola Belajar dan Kelulusan Mahasiswa
    """
    np.random.seed(42)
    n_students = 200
    
    jam_belajar = np.random.uniform(2.0, 25.0, n_students)
    kehadiran = np.random.uniform(50.0, 100.0, n_students)
    nilai_praktikum = np.random.uniform(40.0, 100.0, n_students)
    
    # Label Klasifikasi (1 = Lulus Tepat Waktu, 0 = Terlambat)
    status_lulus = np.where((jam_belajar > 10) & (kehadiran > 75) & (nilai_praktikum > 65), 1, 0)
    
    df = pd.DataFrame({
        'Jam_Belajar': jam_belajar,
        'Kehadiran': kehadiran,
        'Nilai_Praktikum': nilai_praktikum,
        'Status_Lulus': status_lulus
    })
    return df

def main():
    print("=" * 75)
    print("PERTEMUAN 4: Introduction to Machine Learning - Taniza Tajudin")
    print("IMPLEMENTASI TUGAS 2: SUPERVISED vs UNSUPERVISED LEARNING")
    print("=" * 75)
    
    # 1. LOAD DATA
    df = generate_student_ml_data()
    print("\n[1] 5 Data Pertama Mahasiswa (Raw Data):")
    print(df.head())
    
    # PREPARATION & SCALING
    X = df[['Jam_Belajar', 'Kehadiran', 'Nilai_Praktikum']]
    y = df['Status_Lulus']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -------------------------------------------------------------------------
    # 2. SUPERVISED LEARNING (DECISION TREE CLASSIFICATION)
    # -------------------------------------------------------------------------
    print("\n[2] --- SUPERVISED LEARNING (Klasifikasi Kelulusan) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Akurasi Model Decision Tree: {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi Supervised:")
    print(classification_report(y_test, y_pred))
    
    # -------------------------------------------------------------------------
    # 3. UNSUPERVISED LEARNING (K-MEANS CLUSTERING)
    # -------------------------------------------------------------------------
    print("\n[3] --- UNSUPERVISED LEARNING (K-Means Clustering) ---")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Cluster_Belajar'] = kmeans.fit_predict(X_scaled)
    
    print("Hasil Segmentasi Otomatis Mahasiswa (Tanpa Label):")
    for cluster_id in range(2):
        total_mhs = (df['Cluster_Belajar'] == cluster_id).sum()
        avg_jam = df[df['Cluster_Belajar'] == cluster_id]['Jam_Belajar'].mean()
        avg_kehadiran = df[df['Cluster_Belajar'] == cluster_id]['Kehadiran'].mean()
        print(f" -> Cluster {cluster_id}: {total_mhs} Mahasiswa | Rata-rata Belajar: {avg_jam:.1f} jam/minggu | Rata-rata Kehadiran: {avg_kehadiran:.1f}%")
        
    print("\n" + "=" * 75)
    print("[4] SIMULASI INFERENSI MAHASISWA BARU (SUPERVISED PREDICTION):")
    # Contoh: Jam belajar 15 jam, Kehadiran 90%, Nilai Praktikum 80
    mhs_baru = scaler.transform([[15, 90, 80]])
    pred_baru = clf.predict(mhs_baru)
    status_str = "LULUS TEPAT WAKTU" if pred_baru[0] == 1 else "TERLAMBAT"
    print(f" Input: Jam Belajar 15j | Kehadiran 90% | Nilai 80")
    print(f" Hasil Prediksi Model ML: {status_str}")
    print("=" * 75)

if __name__ == "__main__":
    main()