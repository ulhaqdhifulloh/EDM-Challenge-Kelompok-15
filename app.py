# app.py

import streamlit as st
st.set_page_config(page_title="Prediksi Performa Cabang", layout="wide")

import pandas as pd
import joblib
import json
import plotly.graph_objects as go

# --- Fungsi untuk memuat model dan objek lain ---
# Menggunakan cache agar tidak load ulang setiap kali ada interaksi
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_rf.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    with open('store_locations.json', 'r') as f:
        store_locations = json.load(f)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, label_encoder, store_locations, feature_names

# Muat semua file yang dibutuhkan
try:
    model, scaler, le, store_locations, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("File model atau objek pendukung tidak ditemukan. Jalankan skrip `prepare_model.py` terlebih dahulu.")
    st.stop()


# --- Tampilan Aplikasi Streamlit ---

st.title("📊 Prediksi Performa Cabang Coffee Shop")
st.markdown("Aplikasi ini memprediksi apakah sebuah cabang akan menjadi *High*, *Medium*, atau *Low Performer* di bulan berikutnya berdasarkan data operasional bulan ini.")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Data Cabang Bulan Ini")

# Pilihan sumber data
data_source = st.sidebar.radio("Pilih sumber data:", ("Upload Dataset Sendiri", "Gunakan Data Contoh"))

if data_source == "Upload Dataset Sendiri":
    uploaded_file = st.sidebar.file_uploader("Upload file dataset (.xlsx atau .csv)", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_data = pd.read_csv(uploaded_file)
        else:
            df_data = pd.read_excel(uploaded_file)
        st.sidebar.success("Dataset berhasil diupload!")
        st.write("Preview data yang diupload:")
        st.dataframe(df_data.head())
    else:
        df_data = None
else:
    # Data contoh
    df_data = pd.read_excel("Dataset/Coffee Shop Sales.xlsx")
    st.sidebar.info("Menggunakan data contoh bawaan.")
    st.write("Preview data contoh:")
    st.dataframe(df_data.head())

# Fungsi input manual tetap ada, user bisa isi manual atau gunakan data dari df_data

def user_input_features():
    location = st.sidebar.selectbox("📍 Lokasi Cabang", options=store_locations)
    total_sales_amount = st.sidebar.number_input("💰 Total Omzet ($)", min_value=0.0, value=5000.0, step=100.0)
    total_transactions = st.sidebar.number_input("🧾 Jumlah Transaksi", min_value=0, value=1000, step=10)
    total_products_sold = st.sidebar.number_input("☕ Total Produk Terjual", min_value=0, value=2000, step=10)
    unique_product_count = st.sidebar.number_input("📦 Jumlah Produk Unik Terjual", min_value=0, value=50, step=1)
    active_days = st.sidebar.slider("🗓️ Jumlah Hari Aktif dalam Sebulan", min_value=1, max_value=31, value=30)
    
    # Hitung fitur turunan
    avg_transaction_value = total_sales_amount / total_transactions if total_transactions > 0 else 0
    avg_qty_per_transaction = total_products_sold / total_transactions if total_transactions > 0 else 0

    data = {
        'total_sales_amount': total_sales_amount,
        'total_transactions': total_transactions,
        'avg_transaction_value': avg_transaction_value,
        'avg_qty_per_transaction': avg_qty_per_transaction,
        'total_products_sold': total_products_sold,
        'unique_product_count': unique_product_count,
        'active_days': active_days,
        'store_location': location
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan ringkasan input
st.subheader("📝 Ringkasan Input Anda")
col1, col2, col3 = st.columns(3)
col1.metric("Lokasi", input_df['store_location'].iloc[0])
col2.metric("Total Omzet", f"${input_df['total_sales_amount'].iloc[0]:,.2f}")
col3.metric("Jumlah Transaksi", f"{input_df['total_transactions'].iloc[0]}")


# Tombol untuk prediksi
if st.sidebar.button("🚀 Prediksi Sekarang!"):
    
    # --- Proses Prediksi ---
    # 1. One-Hot Encode input user
    input_encoded = pd.get_dummies(input_df, columns=['store_location'])
    
    # 2. Pastikan semua kolom dari training ada di input
    # Kolom yang tidak ada di input (karena lokasi tidak dipilih) akan diisi 0
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # 3. Scaling fitur
    input_scaled = scaler.transform(input_aligned)
    
    # 4. Lakukan prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # 5. Decode hasil prediksi
    predicted_label = le.inverse_transform(prediction)[0]
    
    # --- Tampilkan Hasil Prediksi ---
    st.subheader("📈 Hasil Prediksi")
    
    # Tentukan warna dan emoji berdasarkan hasil
    if predicted_label == 'High':
        color = "green"
        emoji = "🏆"
    elif predicted_label == 'Medium':
        color = "orange"
        emoji = "👍"
    else:
        color = "red"
        emoji = "⚠️"
        
    # Tampilkan hasil dengan gaya
    st.markdown(f"""
    <div style="background-color: #f0f2f6; border-left: 10px solid {color}; padding: 20px; border-radius: 5px;">
        <h2 style="color: {color};">{emoji} Prediksi Performa Bulan Depan: <strong>{predicted_label.upper()} Performer</strong></h2>
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan probabilitas
    st.subheader("Confidence Score (Probabilitas)")
    
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    
    fig = go.Figure()
    colors = ['red', 'orange', 'green'] # Low, Medium, High
    
    for i, class_name in enumerate(le.classes_):
        fig.add_trace(go.Bar(
            y=['Probabilitas'],
            x=[proba_df[class_name].iloc[0]],
            name=f"{class_name} Performer",
            orientation='h',
            marker_color=colors[i],
            text=f"{proba_df[class_name].iloc[0]:.2%}",
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack', 
        title_text='Distribusi Probabilitas Prediksi',
        xaxis_title="Probabilitas",
        yaxis_title="",
        legend_title="Kelas Performa",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)