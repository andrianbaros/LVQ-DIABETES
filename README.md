---

# Klasifikasi Diabetes Menggunakan LVQ

Selamat datang di proyek "Klasifikasi Diabetes Menggunakan LVQ"! Proyek ini bertujuan untuk menggunakan algoritma Learning Vector Quantization (LVQ) untuk mengklasifikasikan data diabetes, menggunakan Streamlit sebagai antarmuka pengguna.

## Table of Contents

1. [Pengenalan Anggota](#pengenalan-anggota)
2. [Fitur](#fitur)
3. [Cara Menjalankan Aplikasi](#cara-menjalankan-aplikasi)
   - [Secara Lokal](#1-menjalankan-aplikasi-secara-lokal)
   - [Secara Online](#2-mengakses-aplikasi-secara-online)
4. [Struktur Proyek](#struktur-proyek)
5. [Informasi Tambahan](#informasi-tambahan)

## Pengenalan Anggota

**KELOMPOK 7**

- **Andrian Baros** (10122003)
- **M. Fathi Zaidan** (10122017)
- **Khotibul Umam** (10122036)
- **Arya Ababil** (10122506)

## Fitur

- **Visualisasi Data**: Menampilkan dataset diabetes dan statistik deskriptif.
- **Training LVQ**: Mengatur parameter learning rate dan jumlah epoch untuk melatih model LVQ.
- **Evaluasi Model**: Menyediakan akurasi model setelah pelatihan.
- **Prediksi**: Membuat prediksi berdasarkan input pengguna.

## Cara Menjalankan Aplikasi

### 1. Menjalankan Aplikasi Secara Lokal

Untuk menjalankan aplikasi di komputer lokal Anda, ikuti langkah-langkah berikut:

1. **Instalasi Dependencies**:
   - Pastikan Python terinstal di sistem Anda.
   - (Opsional) Buat dan aktifkan environment virtual:
     ```bash
     python -m venv venv
     source venv/bin/activate  # Di Windows: venv\Scripts\activate
     ```
   - Instal dependensi yang diperlukan dengan `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

2. **Menjalankan Aplikasi**:
   - Navigasikan ke direktori yang berisi file `lvq.py`:
     ```bash
     cd path/to/your/project
     ```
   - Jalankan aplikasi dengan Streamlit:
     ```bash
     streamlit run lvq.py
     ```
   - Aplikasi akan terbuka secara otomatis di browser. Jika tidak, buka [http://localhost:8501](http://localhost:8501).

### 2. Mengakses Aplikasi Secara Online

Aplikasi ini juga tersedia secara online. Anda dapat mengaksesnya menggunakan link berikut:

- [Klasifikasi Diabetes Menggunakan LVQ - Kelompok 7](https://kelompok7-lvq-diabetes.streamlit.app/)

## Struktur Proyek

- `lvq.py`: Skrip utama untuk aplikasi Streamlit dan implementasi algoritma LVQ.
- `requirements.txt`: Daftar paket Python yang diperlukan untuk menjalankan aplikasi.
- `catatan.txt`: Catatan penggunaan dan petunjuk tambahan.

## Informasi Tambahan

- **Source Code**: Lihat [source code di GitHub](https://github.com/andrianbaros/LVQ-DIABETES).

---

Anda dapat membuat file `README.md` dengan menyalin teks di atas ke editor teks dan menyimpannya sebagai `README.md`.
