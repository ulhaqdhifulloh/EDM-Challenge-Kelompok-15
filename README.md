# EDM-Challenge-Kelompok-15

## Deskripsi Proyek
Proyek ini merupakan solusi untuk tantangan EDM (Exploratory Data Mining) yang berfokus pada analisis dan prediksi penjualan di sebuah coffee shop. Proyek ini mencakup proses EDA (Exploratory Data Analysis), pembuatan model machine learning, serta aplikasi prediksi berbasis Python.

## Struktur Folder & File Penting
- `app.py` : Aplikasi utama untuk melakukan prediksi penjualan.
- `prepare_model.py` : Script untuk pelatihan dan penyimpanan model machine learning.
- `Dataset/Coffee Shop Sales.xlsx` : Dataset utama yang digunakan untuk analisis dan pelatihan model.
- `EDA.ipynb` : Notebook untuk eksplorasi data dan analisis awal.
- `Sales_Forecasting.ipynb` : Notebook untuk proses forecasting penjualan.
- `model.joblib`, `model_rf.joblib` : File model machine learning yang sudah dilatih.
- `scaler.joblib` : File scaler untuk preprocessing fitur numerik.
- `label_encoder.joblib` : Encoder untuk fitur kategorikal.
- `feature_names.json` : Daftar nama fitur yang digunakan pada model.
- `store_locations.json` : Data lokasi toko yang digunakan dalam analisis.

## Cara Instalasi & Menjalankan Aplikasi
1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd EDM-Challenge-Kelompok-15
   ```
2. **Install dependencies**
   Pastikan Python 3.x sudah terinstall. Install library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
   Jika file `requirements.txt` belum tersedia, install manual:
   ```bash
   pip install pandas scikit-learn joblib openpyxl
   ```
3. **Menjalankan aplikasi prediksi**
   ```bash
   python app.py
   ```

## Penjelasan Model & Pipeline
- Model yang digunakan adalah Random Forest dan model lain yang disimpan pada file `.joblib`.
- Pipeline preprocessing mencakup scaling fitur numerik dan encoding fitur kategorikal.
- Notebook EDA dan forecasting dapat digunakan untuk eksplorasi dan pengembangan lebih lanjut.

## Kontributor
- Kelompok 15 EDM Challenge

---
Silakan hubungi anggota kelompok untuk pertanyaan lebih lanjut atau kontribusi pada proyek ini.