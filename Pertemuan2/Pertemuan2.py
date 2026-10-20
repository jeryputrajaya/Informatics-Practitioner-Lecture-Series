"""
================================================================================
PERTEMUAN 2: Bridging Global Infrastructure to Address Local Challenge with HPC
Pemateri   : Asst. Prof. Dr. Worawan Diaz Carballo
Tugas      : Simulasi Pemrosesan Data Bencana Paralel (HPC Benchmark)
Program Studi Informatika - Universitas Bengkulu
================================================================================
"""

import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

def generate_sensor_data(n_samples=500000):
    """
    PERTEMUAN 2 - DATA UNDERSTANDING:
    Membuat Simulasi Data Sensor Cuaca Lokal dalam Jumlah Besar (Big Data)
    """
    np.random.seed(42)
    suhu = np.random.uniform(20.0, 38.0, n_samples)
    kelembapan = np.random.uniform(50.0, 98.0, n_samples)
    kecepatan_angin = np.random.uniform(0.0, 80.0, n_samples)
    intensitas_hujan = np.random.uniform(0.0, 150.0, n_samples)
    
    return pd.DataFrame({
        'Suhu': suhu,
        'Kelembapan': kelembapan,
        'Kecepatan_Angin': kecepatan_angin,
        'Intensitas_Hujan': intensitas_hujan
    })

def hitung_indeks_risiko(row):
    """
    Fungsi Komputasi Numerik Kompleks (Simulasi Tugas HPC)
    Menghitung Indeks Potensi Bencana berdasarkan Parameter Cuaca
    """
    # Formula simulasi beban komputasi berat
    val = (row[0] * 0.2) + (row[1] * 0.3) + (row[2] * 0.1) + (row[3] * 0.4)
    res = np.sin(val) ** 2 + np.cos(val) ** 2  # Operasi matematika floating-point
    return res

def proses_sekuensial(df):
    """Proses Komputasi Konvensional (Single-Core / Tanpa HPC)"""
    return [hitung_indeks_risiko(row) for row in df.values]

def proses_paralel_chunk(chunk):
    """Proses Komputasi Paralel pada Setiap Chunk Data"""
    return [hitung_indeks_risiko(row) for row in chunk]

def main():
    print("=" * 75)
    print("PERTEMUAN 2: Bridging Global Infrastructure to Address Local Challenge with HPC")
    print("IMPLEMENTASI TUGAS 2: SIMULASI KOMPUTASI PARALEL (HPC BENCHMARK)")
    print("=" * 75)
    
    n_samples = 300000
    print(f"\n[1] Memuat dataset cuaca simulasi ({n_samples:,} baris data)...")
    df = generate_sensor_data(n_samples)
    print(df.head())
    
    # --- 1. PROSES SEKUANSEIAL (SINGLE CORE) ---
    print("\n[2] Menjalankan pemrosesan SEKUANSIAL (1 CPU Core)...")
    start_time = time.time()
    res_seq = proses_sekuensial(df)
    time_seq = time.time() - start_time
    print(f"-> Waktu Eksekusi Sekuensial: {time_seq:.4f} detik")
    
    # --- 2. PROSES PARALEL (HPC MULTI-CORE) ---
    num_cores = cpu_count()
    print(f"\n[3] Menjalankan pemrosesan PARALEL HPC (Menggunakan {num_cores} CPU Cores)...")
    
    # Membagi data menjadi beberapa bagian sesuai jumlah core CPU
    chunks = np.array_split(df.values, num_cores)
    
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        res_parallel_chunks = pool.map(proses_paralel_chunk, chunks)
    
    # Menggabungkan hasil komputasi dari seluruh core
    res_parallel = [item for sublist in res_parallel_chunks for item in sublist]
    time_parallel = time.time() - start_time
    
    print(f"-> Waktu Eksekusi Paralel (HPC): {time_parallel:.4f} detik")
    
    # --- 3. EVALUASI KINERJA (SPEEDUP) ---
    speedup = time_seq / time_parallel
    efisiensi = (speedup / num_cores) * 100
    
    print("\n" + "=" * 75)
    print("[4] HASIL BENCHMARK PERFORMA HPC:")
    print(f" Jumlah CPU Core Digunakan : {num_cores} Core")
    print(f" Akselerasi Kecepatan (Speedup): {speedup:.2f}x Lebih Cepat")
    print(f" Efisiensi Efektif Komputasi  : {efisiensi:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()