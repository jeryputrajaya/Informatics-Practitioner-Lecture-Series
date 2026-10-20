"""
================================================================================
PERTEMUAN 5: Human-Computer Interaction & Data Visualization
Pemateri   : Arafat Febriandirza
Tugas      : Implementasi Visualisasi Data dan Dasbor Analitik Interaktif (HCI)
Program Studi Informatika - Universitas Bengkulu
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
# Menggunakan backend Agg agar tidak membutuhkan jendela GUI interaktif yang rawan error
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_hci_dashboard_data():
    """
    PERTEMUAN 5 - DATA UNDERSTANDING:
    Membuat Dataset Simulasi Interaksi Pengguna dan Performa Sistem
    """
    bulan = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    np.random.seed(42)
    
    pengguna_aktif = np.array([1200, 1350, 1500, 1800, 2100, 2500, 2900, 3200, 3800, 4200, 4700, 5100])
    waktu_respon = np.random.uniform(120.0, 350.0, len(bulan))  # ms
    tingkat_kepuasan = np.random.uniform(3.8, 4.9, len(bulan))  # skala 1-5
    
    df = pd.DataFrame({
        'Bulan': bulan,
        'Pengguna_Aktif': pengguna_aktif,
        'Waktu_Respon_ms': waktu_respon,
        'Tingkat_Kepuasan': tingkat_kepuasan
    })
    return df

def main():
    print("=" * 75)
    print("PERTEMUAN 5: Human-Computer Interaction & Data Visualization - Arafat Febriandirza")
    print("IMPLEMENTASI TUGAS 2: RANCANGAN DASBOR ANALITIK INTERAKTIF (HCI)")
    print("=" * 75)
    
    # 1. LOAD DATA
    df = generate_hci_dashboard_data()
    print("\n[1] Ringkasan Data Interaksi dan Performa Sistem:")
    print(df.head())
    
    # 2. IMPLEMENTASI PRINSIP VISUALISASI DATA (MATPLOTLIB & SEABORN)
    print("\n[2] Menghasilkan Dasbor Visualisasi Data (Prinsip HCI)...")
    
    # Mengatur gaya tampilan visual (Clean & User-Centered)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DASBOR PERFORMA INTERAKSI PENGGUNA & APLIKASI (PERTEMUAN 5)', fontsize=14, fontweight='bold')
    
    # GRAFIK 1: Tren Pertumbuhan Pengguna Aktif (Line Chart)
    sns.lineplot(ax=axes[0], data=df, x='Bulan', y='Pengguna_Aktif', marker='o', color='#1f77b4', linewidth=2.5)
    axes[0].set_title('Pertumbuhan Pengguna Aktif Bulanan (DAU)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Bulan')
    axes[0].set_ylabel('Jumlah Pengguna Aktif')
    
    # GRAFIK 2: Waktu Respon Sistem (Bar Chart)
    sns.barplot(ax=axes[1], data=df, x='Bulan', y='Waktu_Respon_ms', color='#2ca02c')
    axes[1].set_title('Rata-rata Waktu Respon Sistem (ms)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Bulan')
    axes[1].set_ylabel('Waktu Respon (ms)')
    
    plt.tight_layout()
    
    # Menyimpan file gambar langsung ke folder Pertemuan5
    output_image = "dasbor_hci_visualization.png"
    plt.savefig(output_image, dpi=300)
    print(f" -> Grafik visualisasi berhasil disimpan sebagai file '{output_image}'")
    plt.close()
    
    print("\n" + "=" * 75)
    print("[3] KESIMPULAN VISUALISASI DASBOR:")
    print(f" Total Pertumbuhan Pengguna : {df['Pengguna_Aktif'].iloc[-1] - df['Pengguna_Aktif'].iloc[0]} Pengguna Baru")
    print(f" Rata-rata Kepuasan CSAT  : {df['Tingkat_Kepuasan'].mean():.2f} / 5.0")
    print("=" * 75)

if __name__ == "__main__":
    main()