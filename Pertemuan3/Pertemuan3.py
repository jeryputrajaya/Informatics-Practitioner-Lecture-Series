"""
================================================================================
PERTEMUAN 3: Career and Competition in Industrial Era with GIS
Pemateri   : Widi Primadepa (City Care Property, Selandia Baru)
Tugas      : Implementasi Sistem Informasi Geografis & Analisis Spasial Aset
Program Studi Informatika - Universitas Bengkulu
================================================================================
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point

def generate_gis_asset_data():
    """
    PERTEMUAN 3 - DATA UNDERSTANDING:
    Membuat Data Spasial Aset Kota (Latitude, Longitude & Atribut Aset)
    """
    np.random.seed(42)
    n_assets = 10
    
    # Koordinat simulasi sekitar wilayah Bengkulu/Christchurch
    latitudes = np.random.uniform(-3.8000, -3.7500, n_assets)
    longitudes = np.random.uniform(102.2500, 102.3000, n_assets)
    
    nama_aset = [
        "Taman Kota Remaja", "Perpustakaan Daerah", "Stadion Olahraga",
        "Taman Budaya", "Hutan Kota", "Alun-Alun Fasilitas",
        "Taman Lansia", "Area Bermain Anak", "Kawasan Hijau Mandiri", "Taman Rekreasi"
    ]
    
    kondisi_aset = np.random.randint(40, 100, n_assets)
    jumlah_pohon = np.random.randint(10, 150, n_assets)
    
    df = pd.DataFrame({
        'ID_Aset': [f"AST-{1001+i}" for i in range(n_assets)],
        'Nama_Aset': nama_aset,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Kondisi_Fisik': kondisi_aset,
        'Jumlah_Pohon': jumlah_pohon
    })
    return df

def hitung_jarak_spasial(lat1, lon1, lat2, lon2):
    """Fungsi Analisis Jarak Geospasial Sederhana (Spatial Proximity)"""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0  # Konversi derajat ke km (estimasi)

def main():
    print("=" * 75)
    print("PERTEMUAN 3: Career and Competition in Industrial Era with GIS - Widi Primadepa")
    print("IMPLEMENTASI TUGAS 2: ANALISIS SPASIAL & MANAJEMEN ASET KOTA")
    print("=" * 75)
    
    # 1. LOAD DATA ASET SPASIAL
    df_aset = generate_gis_asset_data()
    print("\n[1] Data Spasial Aset dan Atribut Kota:")
    print(df_aset[['ID_Aset', 'Nama_Aset', 'Latitude', 'Longitude', 'Kondisi_Fisik']].head())
    
    # 2. PEMBUATAN OBJEK GEOMETRI SPASIAL (SHAPELY POINT)
    df_aset['Geometry'] = [Point(xy) for xy in zip(df_aset['Longitude'], df_aset['Latitude'])]
    
    # 3. SPATIAL ANALYSIS: Menentukan Aset yang Membutuhkan Pemeliharaan
    # Kriteria: Kondisi Fisik < 70 (Perlu Tindakan Solver / Maintenance)
    print("\n[2] Analisis Data-Driven: Filter Aset Berprioritas Pemeliharaan (Kondisi < 70):")
    aset_butuh_perbaikan = df_aset[df_aset['Kondisi_Fisik'] < 70]
    
    for index, row in aset_butuh_perbaikan.iterrows():
        print(f" -> [{row['ID_Aset']}] {row['Nama_Aset']} | Skor Kondisi: {row['Kondisi_Fisik']}/100 -> [STATUS: PERLU REHABILITASI]")
    
    # 4. SPATIAL PROXIMITY ANALYSIS: Mengukur Jarak Antara Aset 1 dan Aset Lainnya
    ref_asset = df_aset.iloc[0]
    print(f"\n[3] Analisis Jangkauan Spasial dari Pusat Aset '{ref_asset['Nama_Aset']}':")
    
    for i in range(1, 5):
        target_asset = df_aset.iloc[i]
        jarak = hitung_jarak_spasial(
            ref_asset['Latitude'], ref_asset['Longitude'],
            target_asset['Latitude'], target_asset['Longitude']
        )
        print(f" -> Jarak ke {target_asset['Nama_Aset']} ({target_asset['ID_Aset']}): {jarak:.2f} km")
        
    print("\n" + "=" * 75)
    print("[4] KESIMPULAN SISTEM ASET GIS:")
    print(f" Total Aset Dipetakan  : {len(df_aset)} Titik Lokasi Spasial")
    print(f" Aset Membutuhkan Action: {len(aset_butuh_perbaikan)} Lokasi")
    print("=" * 75)

if __name__ == "__main__":
    main()