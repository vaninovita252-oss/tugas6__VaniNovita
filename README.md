# Deteksi Tanda Tangan pada Citra Dokumen

Proyek ini merupakan implementasi **Pengolahan Citra Digital (PCD)** untuk mendeteksi keberadaan tanda tangan pada citra dokumen menggunakan Python dan OpenCV.

Sistem melakukan beberapa tahapan pengolahan citra, mulai dari konversi grayscale, thresholding, operasi morfologi, segmentasi, hingga klasifikasi untuk menentukan apakah suatu citra memiliki tanda tangan (**SIGNATURE PRESENT**) atau tidak (**SIGNATURE ABSENT**).

## Fitur

* **Grayscale:** Mengubah citra berwarna menjadi citra keabuan.
* **Global Threshold:** Memisahkan objek berdasarkan nilai ambang tetap.
* **Adaptive Threshold:** Menentukan nilai ambang berdasarkan kondisi lokal citra.
* **Perbandingan Thresholding:** Menghitung skor kualitas untuk membandingkan kedua metode.
* **Morphological Opening:** Mengurangi noise dan objek kecil.
* **Morphological Closing:** Menutup celah kecil pada objek.
* **Segmentasi:** Menghasilkan citra biner sebagai hasil pengolahan.
* **Klasifikasi:** Menggunakan foreground ratio untuk menentukan keberadaan tanda tangan.
* **Visualisasi:** Menampilkan hasil setiap tahapan pengolahan dalam gambar perbandingan.
* **Laporan Analisis:** Menyimpan hasil analisis dan pengujian dalam file teks.

## Teknologi yang Digunakan

* Python
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* pathlib

## Struktur Folder

```text
Deteksi-Tanda-Tangan/
│
├── data/
│   ├── SIGNATURE PRESENT/
│   │   ├── citra1.png
│   │   ├── citra2.png
│   │   └── ...
│   │
│   └── SIGNATURE ABSENT/
│       ├── citra1.png
│       ├── citra2.png
│       └── ...
│
├── results/
│   ├── 01_GRAYSCALE/
│   ├── 02_GLOBAL_THRESHOLD/
│   ├── 03_ADAPTIVE_THRESHOLD/
│   ├── 04_COMPARISON/
│   ├── 05_OPENING/
│   ├── 06_CLOSING/
│   ├── 07_SEGMENTATION/
│   ├── 08_ANALYSIS/
│   │   └── hasil_analisis.txt
│   │
│   └── 09_CLASSIFICATION/
│       └── hasil_klasifikasi.txt
│
├── main.py
└── README.md
```

## Alur Pengolahan Citra

Sistem menjalankan tahapan berikut:

1. **Input Citra:** Membaca citra dari folder SIGNATURE PRESENT dan SIGNATURE ABSENT.
2. **Grayscale:** Mengubah citra RGB/BGR menjadi citra grayscale.
3. **Global Threshold:** Melakukan thresholding menggunakan nilai ambang tetap.
4. **Adaptive Threshold:** Melakukan thresholding berdasarkan kondisi intensitas lokal.
5. **Perbandingan:** Menghitung skor kualitas hasil kedua metode dan memilih metode dengan skor lebih tinggi.
6. **Morphological Opening:** Mengurangi komponen kecil pada hasil thresholding terpilih.
7. **Morphological Closing:** Menutup celah kecil pada hasil Opening.
8. **Segmentasi:** Menghasilkan citra hasil pemisahan foreground dan background.
9. **Perhitungan Foreground Ratio:** Menghitung persentase piksel foreground terhadap seluruh piksel citra.
10. **Klasifikasi:** Menentukan hasil klasifikasi berdasarkan nilai foreground ratio.

## Parameter

Parameter yang digunakan pada program:

| Parameter                | Nilai | Keterangan                    |
| ------------------------ | ----: | ----------------------------- |
| Global Threshold         |   127 | Nilai ambang tetap            |
| Adaptive Block Size      |    11 | Ukuran area lokal             |
| Adaptive C               |     2 | Konstanta pengurang threshold |
| Kernel Morphology        | 3 × 3 | Ukuran kernel morfologi       |
| Classification Threshold |    2% | Ambang klasifikasi            |

Aturan klasifikasi:

* Foreground Ratio ≥ 2%: **SIGNATURE PRESENT**
* Foreground Ratio < 2%: **SIGNATURE ABSENT**

Nilai ambang tersebut merupakan parameter eksperimen dan dapat disesuaikan berdasarkan hasil pengujian.

## Instalasi

Pastikan Python telah terpasang pada komputer.

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/NAMA-REPOSITORY.git
```

Masuk ke direktori proyek:

```bash
cd NAMA-REPOSITORY
```

Ganti `USERNAME` dan `NAMA-REPOSITORY` sesuai dengan akun serta nama repository GitHub kamu.

### 2. Instal Library

Jalankan perintah berikut pada terminal:

```bash
pip install opencv-python numpy matplotlib
```

### 3. Siapkan Dataset

Masukkan citra dokumen ke folder sesuai kategori:

* `data/SIGNATURE PRESENT/` untuk citra yang memiliki tanda tangan.
* `data/SIGNATURE ABSENT/` untuk citra yang tidak memiliki tanda tangan.

Format citra yang didukung:

* JPG
* JPEG
* PNG
* BMP
* TIF
* TIFF

### 4. Jalankan Program

Jalankan perintah berikut:

```bash
python analysis_classification.py
```

Program akan memproses seluruh citra yang ditemukan pada kedua folder dataset dan menyimpan hasilnya di folder `results/`.

## Hasil Pengolahan

Hasil pengolahan disimpan berdasarkan tahapan proses.

| Folder                  | Hasil                                       |
| ----------------------- | ------------------------------------------- |
| `01_GRAYSCALE`          | Citra grayscale                             |
| `02_GLOBAL_THRESHOLD`   | Hasil Global Threshold                      |
| `03_ADAPTIVE_THRESHOLD` | Hasil Adaptive Threshold                    |
| `04_COMPARISON`         | Visualisasi perbandingan tahapan pengolahan |
| `05_OPENING`            | Hasil operasi Opening                       |
| `06_CLOSING`            | Hasil operasi Closing                       |
| `07_SEGMENTATION`       | Hasil segmentasi                            |
| `08_ANALYSIS`           | Laporan analisis dan evaluasi               |
| `09_CLASSIFICATION`     | Hasil klasifikasi setiap citra              |

File `hasil_analisis.txt` berisi informasi tahapan pengolahan, parameter, hasil analisis, ringkasan pengujian, akurasi, dan penjelasan metode.

File `hasil_klasifikasi.txt` berisi label aktual, metode threshold terpilih, foreground pixel, foreground ratio, prediksi, dan status klasifikasi setiap citra.

## Evaluasi

Hasil klasifikasi dievaluasi dengan membandingkan prediksi sistem terhadap label aktual dataset.

Status pengujian terdiri atas:

* **CORRECT:** Prediksi sesuai dengan label aktual.
* **INCORRECT:** Prediksi tidak sesuai dengan label aktual.

Akurasi dihitung menggunakan rumus:

$$
\text{Akurasi}=\frac{\text{Jumlah Prediksi Benar}}{\text{Total Citra}}\times100\%
$$

Nilai akurasi menunjukkan hasil pengujian pada dataset yang digunakan. Hasil tersebut tidak secara otomatis menggambarkan kinerja sistem pada semua jenis dokumen.

## Catatan

Sistem ini menggunakan foreground ratio sebagai dasar klasifikasi. Oleh karena itu, noise, tekstur kertas, kualitas citra, pencahayaan, dan hasil thresholding dapat memengaruhi hasil deteksi.

Citra tanpa tanda tangan yang memiliki banyak noise dapat menghasilkan foreground ratio tinggi sehingga berpotensi salah diklasifikasikan. Parameter dan metode perlu dievaluasi lebih lanjut menggunakan dataset yang beragam.

## Tujuan Proyek

Proyek ini dibuat sebagai implementasi pembelajaran Pengolahan Citra Digital, khususnya penerapan:

* Konversi grayscale.
* Teknik thresholding.
* Operasi morfologi.
* Segmentasi citra.
* Analisis foreground.
* Klasifikasi sederhana pada citra dokumen.

---

**Pengembangan:** Vani Novita
**Program Studi:** Ilmu Komputer
**Fakultas:** Matematika dan Ilmu Pengetahuan Alam
**Universitas:** Universitas Halu Oleo
**Tahun:** 2026
