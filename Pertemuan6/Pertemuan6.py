"""
================================================================================
PERTEMUAN 6: Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI
Tugas      : Klasifikasi Penyebab Deforestasi Berbasis Citra Satelit & AI
Program Studi Informatika - Universitas Bengkulu
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_satellite_deforestation_data():
    """
    PERTEMUAN 6 - DATA UNDERSTANDING:
    Membuat Dataset Simulasi Ekstraksi Fitur Citra Satelit (NDVI, NDWI, NBR)
    """
    np.random.seed(42)
    n_samples = 300
    
    # Nilai indeks spektral citra
    ndvi = np.random.uniform(0.1, 0.8, n_samples)
    ndwi = np.random.uniform(-0.5, 0.4, n_samples)
    nbr = np.random.uniform(-0.6, 0.6, n_samples)
    
    # Logika Penentuan Penyebab (0 = Pertanian Lokal, 1 = Industri Kelapa Sawit, 2 = Kebakaran Hutan)
    pemicu = []
    for i in range(n_samples):
        if nbr[i] < -0.2:
            pemicu.append(2)  # Kebakaran Hutan
        elif ndvi[i] > 0.5 and ndwi[i] > 0.0:
            pemicu.append(1)  # Industri Kelapa Sawit
        else:
            pemicu.append(0)  # Pertanian Swadaya / Lainnya
            
    df = pd.DataFrame({
        'NDVI': ndvi,
        'NDWI': ndwi,
        'NBR': nbr,
        'Penyebab_Deforestasi': pemicu
    })
    return df

def main():
    print("=" * 75)
    print("PERTEMUAN 6: Attributing Direct Drivers to Deforestation Alerts with AI")
    print("IMPLEMENTASI TUGAS 2: KLASIFIKASI PEMICU DEFORESTASI BERBASIS CITRA SATELIT")
    print("=" * 75)
    
    # 1. LOAD DATA
    df = generate_satellite_deforestation_data()
    print("\n[1] 5 Data Pertama Indeks Spektral Citra Satelit:")
    print(df.head())
    
    # PREPARATION & SCALING
    X = df[['NDVI', 'NDWI', 'NBR']]
    y = df['Penyebab_Deforestasi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. MODELING (RANDOM FOREST CLASSIFIER)
    print("\n[2] Melatih Model AI untuk Mengidentifikasi Pemicu Deforestasi...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 3. EVALUATION
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAkurasi Model AI Citra Satelit: {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi Pemicu Deforestasi:")
    target_names = ['Pertanian Swadaya', 'Industri Kelapa Sawit', 'Kebakaran Hutan']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 4. INFERENSI/PREDIKSI DATA DETEKSI SATELIT BARU
    print("=" * 75)
    print("[3] SIMULASI INFERENSI PIKSEL CITRA SATELIT BARU:")
    # Contoh piksel alert baru: NDVI = 0.65, NDWI = 0.15, NBR = 0.2 (Indikasi Sawit)
    piksel_baru = scaler.transform([[0.65, 0.15, 0.2]])
    pred_pemicu = model.predict(piksel_baru)
    print(f" Input Fitur Spektral : NDVI=0.65 | NDWI=0.15 | NBR=0.2")
    print(f" Prediksi Penyebab AI : {target_names[pred_pemicu[0]]}")
    print("=" * 75)

if __name__ == "__main__":
    main()