import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Dashboard Prediksi Kinerja Cabang", layout="wide")

# --- Judul ---
st.title("Dashboard Prediksi Kinerja Cabang")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("best_model_random_forest.joblib")

model = load_model()

# --- Load Data ---
@st.cache_data
def load_data():
    # Assume data is in Dataset/Coffee Shop Sales.xlsx
    df = pd.read_excel(os.path.join("Dataset", "Coffee Shop Sales.xlsx"))
    return df

df = load_data()

# --- Preprocessing (simplified, adjust as needed) ---
def preprocess_data(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['month'] = df['transaction_date'].dt.month
    df['year'] = df['transaction_date'].dt.year
    df['total_sales'] = df['transaction_qty'] * df['unit_price']
    agg = df.groupby(['store_id', 'store_location', 'year', 'month']).agg(
        total_sales_amount=('total_sales', 'sum'),
        total_transactions=('transaction_id', 'count'),
        total_products_sold=('transaction_qty', 'sum'),
        unique_product_count=('product_id', 'nunique'),
        active_days=('transaction_date', lambda date: date.nunique())
    ).reset_index()
    agg['avg_transaction_value'] = agg['total_sales_amount'] / agg['total_transactions']
    agg['avg_qty_per_transaction'] = agg['total_products_sold'] / agg['total_transactions']
    agg.sort_values(['store_id', 'year', 'month'], inplace=True)
    return agg

df_agg = preprocess_data(df)

# --- Sidebar Filter ---
locations = df_agg['store_location'].unique().tolist()
selected_location = st.sidebar.selectbox("Filter lokasi cabang", ["Semua"] + locations)

if selected_location != "Semua":
    df_agg = df_agg[df_agg['store_location'] == selected_location]

# --- Overview Section ---
st.header("Overview Performa Cabang")

# Placeholder for KPI and branch list (to be filled in next steps)
st.write("Ringkasan performa cabang dan daftar cabang akan ditampilkan di sini.")

# --- Detail Section (to be implemented: on branch click) ---
# Placeholder for detail view

# --- Download Section ---
st.sidebar.header("Unduh Laporan")
st.sidebar.download_button(
    label="Download CSV",
    data=df_agg.to_csv(index=False),
    file_name="laporan_performa_cabang.csv",
    mime="text/csv"
)

# PDF download will be implemented in the next step
