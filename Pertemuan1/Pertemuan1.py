"""
PERTEMUAN 1: Make Sense of Data with Analysis and AI - Andria Arisal
Tugas : Studi Kasus & Implementasi Machine Learning (CRISP-DM)
Program Studi Informatika - Universitas Bengkulu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_dummy_data():
    """
    DATA UNDERSTANDING:
    Membuat Dataset Simulasi Kredit Nasabah berdasarkan kriteria keuangan
    """
    np.random.seed(42)
    n_samples = 200
    
    usia = np.random.randint(21, 60, n_samples)
    pendapatan = np.random.randint(3, 25, n_samples) * 1000000  # Dalam Rupiah
    jumlah_hutang = np.random.randint(0, 10, n_samples) * 1000000
    skor_kredit = np.random.randint(300, 850, n_samples)
    
    # Logika penentuan status kredit (1 = Layak, 0 = Tidak Layak)
    status_kredit = np.where((pendapatan > 7000000) & (skor_kredit > 600) & (jumlah_hutang < 5000000), 1, 0)
    
    data = pd.DataFrame({
        'Usia': usia,
        'Pendapatan': pendapatan,
        'Jumlah_Hutang': jumlah_hutang,
        'Skor_Kredit': skor_kredit,
        'Status_Kredit': status_kredit
    })
    return data

def main():
    print("=" * 75)
    print("PERTEMUAN 1: Make Sense of Data with Analysis and AI - Andria Arisal")
    print("IMPLEMENTASI TUGAS : PREDOKSI KELAYAKAN KREDIT NASABAH (CRISP-DM)")
    print("=" * 75)
    
    # Load Data
    df = generate_dummy_data()
    print("\n[1] 5 Data Pertama Nasabah (Raw Data):")
    print(df.head())
    
    # DATA PREPARATION
    X = df.drop('Status_Kredit', axis=1)
    y = df['Status_Kredit']
    
    # Split data: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # MODELING (Machine Learning)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n[2] Hasil Evaluasi Model Machine Learning:")
    print(f"Akurasi Model: {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # INFERENCE / PREDIKSI DATA NASABAH BARU (Prescriptive Analysis)
    # Contoh: Nasabah Usia 28th, Pendapatan 10jt, Hutang 2jt, Skor Kredit 720
    nasabah_baru = np.array([[28, 10000000, 2000000, 720]])
    nasabah_scaled = scaler.transform(nasabah_baru)
    hasil_prediksi = model.predict(nasabah_scaled)
    
    print("\n[3] Simulasi Keputusan Kredit Nasabah Baru:")
    status = "LAYAK (APPROVED)" if hasil_prediksi[0] == 1 else "TIDAK LAYAK (REJECTED)"
    print(f"Hasil Prediksi Sistem AI: {status}")
    print("=" * 75)

if __name__ == "__main__":
    main()