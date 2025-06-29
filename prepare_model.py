# prepare_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json

print("Memulai proses persiapan model...")

# --- 1. Load & Feature Engineering (Sama seperti di notebook) ---
df = pd.read_excel("Dataset/Coffee Shop Sales.xlsx")

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['month'] = df['transaction_date'].dt.month
df['year'] = df['transaction_date'].dt.year
df['total_sales'] = df['transaction_qty'] * df['unit_price']

df_agg = df.groupby(['store_id', 'store_location', 'year', 'month']).agg(
    total_sales_amount=('total_sales', 'sum'),
    total_transactions=('transaction_id', 'count'),
    total_products_sold=('transaction_qty', 'sum'),
    unique_product_count=('product_id', 'nunique'),
    active_days=('transaction_date', lambda date: date.nunique())
).reset_index()

df_agg['avg_transaction_value'] = df_agg['total_sales_amount'] / df_agg['total_transactions']
df_agg['avg_qty_per_transaction'] = df_agg['total_products_sold'] / df_agg['total_transactions']

df_agg.sort_values(['store_id', 'year', 'month'], inplace=True)

# --- 2. Target Variable Creation ---
df_agg['next_month_sales'] = df_agg.groupby('store_id')['total_sales_amount'].shift(-1)
df_agg.dropna(subset=['next_month_sales'], inplace=True)

quantiles = df_agg['next_month_sales'].quantile([0.25, 0.75]).to_dict()
low_threshold = quantiles[0.25]
high_threshold = quantiles[0.75]

def assign_performance_label(sales):
    if sales >= high_threshold:
        return 'High'
    elif sales > low_threshold:
        return 'Medium'
    else:
        return 'Low'

df_agg['performance_category'] = df_agg['next_month_sales'].apply(assign_performance_label)


# --- 3. Preprocessing Final ---
features = [
    'total_sales_amount', 'total_transactions', 'avg_transaction_value',
    'avg_qty_per_transaction', 'total_products_sold', 'unique_product_count',
    'active_days', 'store_location'
]
target = 'performance_category'

X = df_agg[features]
y = df_agg[target]

# Simpan daftar lokasi toko untuk digunakan di aplikasi Streamlit
store_locations = X['store_location'].unique().tolist()
with open('store_locations.json', 'w') as f:
    json.dump(store_locations, f)

# Encode fitur kategori
X_encoded = pd.get_dummies(X, columns=['store_location'], drop_first=True)

# Simpan nama kolom setelah encoding
feature_names = X_encoded.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Encode label target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standarisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Balancing data dengan SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)


# --- 4. Latih dan Simpan Model ---
print("Melatih model Random Forest...")
# Gunakan parameter yang sudah dioptimalkan dari notebook
final_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
final_model.fit(X_resampled, y_resampled)

# Simpan model dan objek preprocessing
joblib.dump(final_model, 'model_rf.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\nProses selesai! File-file berikut telah dibuat:")
print("- model_rf.joblib (Model utama)")
print("- scaler.joblib (Scaler untuk normalisasi data)")
print("- label_encoder.joblib (Encoder untuk label target)")
print("- store_locations.json (Daftar lokasi toko)")
print("- feature_names.json (Daftar nama fitur)")