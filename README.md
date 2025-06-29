# Prospekta: Sistem Prediksi Kinerja Cabang Coffee Shop

## 📝 Deskripsi Proyek

**Prospekta** adalah sistem prediktif berbasis *machine learning* untuk mengklasifikasikan kinerja cabang *coffee shop* di masa mendatang (High, Medium, atau Low Performer). Proyek ini dibangun untuk membantu manajemen dalam mengambil keputusan strategis yang proaktif, seperti alokasi stok dan penempatan staf, guna menghindari inefisiensi dan potensi kerugian. Solusi ini diimplementasikan dalam bentuk *dashboard* interaktif menggunakan Streamlit.

## 👥 Tim Kami

| Nama | NIM |
| :--- | :--- |
| Muhammad Zulfikri Mansur | 1202220139 |
| Dhifulloh Dhiya Ulhaq | 1202220139 |

*Proyek ini merupakan bagian dari Study Group Enterprise Data Management (EDM) G6 - Kelompok 15.*

## 🎯 Permasalahan

Manajemen *coffee shop* kesulitan memprediksi performa penjualan cabang secara akurat, sehingga keputusan bisnis seringkali bersifat reaktif berdasarkan data historis. Hal ini menyebabkan masalah operasional seperti kelebihan stok di cabang sepi atau kekurangan staf di cabang yang ramai.

## 💡 Solusi

Model *machine learning* menganalisis data operasional bulanan (total omzet, jumlah transaksi, dll.) untuk memprediksi kategori performa cabang pada bulan berikutnya. Hasil prediksi disajikan melalui *dashboard* interaktif untuk memudahkan manajemen dalam mengambil tindakan.

## 📊 Dataset

Dataset yang digunakan adalah data transaksi dari **Maven Roasters**, sebuah kedai kopi dengan tiga lokasi di New York City.

  - **Jumlah Entri:** 149.116
  - **Kolom Utama:** `transaction_date`, `transaction_qty`, `store_location`, `unit_price`, `product_category`.

## ⚙️ Metodologi

1.  **Eksplorasi Data (EDA):**

      * Penjualan tertinggi terjadi pada bulan Juni dan terendah pada Februari.
      * Jam sibuk penjualan adalah antara pukul 08:00 hingga 23:00.
      * Kopi adalah kategori produk dengan penjualan tertinggi di semua cabang.

2.  **Pra-pemrosesan Data:**

      * **Feature Engineering:** Membuat kolom baru seperti `total_sales` (`transaction_qty` \* `unit_price`) dan mengekstrak `month`, `day`, serta `time_of_day` dari tanggal transaksi.
      * **Encoding:** Mengubah data kategorikal (`product_category`, `time_of_day`) menjadi numerik menggunakan Label Encoder.

3.  **Pemodelan:**

      * **Algoritma:** `RandomForestRegressor` dari library Scikit-learn.
      * **Parameter Utama:** `n_estimators=100`, `max_depth=10`, `min_samples_leaf=5`.

4.  **Evaluasi Model:**

      * **MAE (Mean Absolute Error):** 0.042
      * **RMSE (Root Mean Squared Error):** 1.788
      * **R² Score:** 0.885 (Model mampu menjelaskan 88.5% variansi data).

## 🚀 Output

Hasil akhir dari proyek ini adalah sebuah *dashboard* interaktif yang dibangun dengan Streamlit. Pengguna dapat memasukkan data operasional cabang (total omzet, jumlah transaksi, dll.) untuk mendapatkan prediksi performa bulan depan beserta tingkat kepercayaan (probabilitas).

## 🛠️ Cara Menjalankan Proyek

1.  **Clone Repository**

    ```bash
    git clone https://github.com/username/repository-name.git
    cd repository-name
    ```

2.  **Buat dan Aktifkan Virtual Environment (Direkomendasikan)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instalasi Dependensi**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi Streamlit**

    ```bash
    streamlit run app.py
    ```

-----