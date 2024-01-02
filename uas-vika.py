import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier

model = pickle.load(open('d3_tingkat_kemiskinan.sav', 'rb'))

st.title('Klasifikasi Tingkat Kemiskinan')

# List provinsi
provinsi_list = [
    'ACEH', 'SUMATERA UTARA', 'SUMATERA BARAT', 'RIAU', 'JAMBI',
    'SUMATERA SELATAN', 'BENGKULU', 'LAMPUNG', 'KEP. BANGKA BELITUNG',
    'KEPULAUAN RIAU', 'DKI JAKARTA', 'JAWA BARAT', 'JAWA TENGAH',
    'D I YOGYAKARTA', 'JAWA TIMUR', 'BANTEN', 'BALI',
    'NUSA TENGGARA BARAT', 'NUSA TENGGARA TIMUR', 'KALIMANTAN BARAT',
    'KALIMANTAN TENGAH', 'KALIMANTAN SELATAN', 'KALIMANTAN TIMUR',
    'KALIMANTAN UTARA', 'SULAWESI UTARA', 'SULAWESI TENGAH',
    'SULAWESI SELATAN', 'SULAWESI TENGGARA', 'GORONTALO',
    'SULAWESI BARAT', 'MALUKU', 'MALUKU UTARA', 'PAPUA BARAT', 'PAPUA'
]

col1, col2 = st.columns(2)

with col1:
    Provinsi = st.selectbox("Pilih Provinsi:", provinsi_list)
    persentase_pm = st.number_input('Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)')
    lama_sekolah = st.number_input("Rata-rata Lama Sekolah Penduduk 15+ (Tahun)")
    pengeluaran_pk = st.number_input("Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)")
    IPM = st.number_input("Indeks Pembangunan Manusia")
    UHH = st.number_input("Umur Harapan Hidup (Tahun)")

with col2:
    sanitasi_layak = st.number_input("Persentase rumah tangga yang memiliki akses terhadap sanitasi layak")
    airminum_layak = st.number_input("Persentase rumah tangga yang memiliki akses terhadap air minum layak")
    TPT = st.number_input("Tingkat Pengangguran Terbuka")
    TPAK = st.number_input("Tingkat Partisipasi Angkatan Kerja")
    PDRB = st.number_input("PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)")

encoder_prov = LabelEncoder()
encoder_prov.fit(provinsi_list)
prov_encoded = encoder_prov.transform([Provinsi])[0]

# Display prediction result and decision tree in the sidebar
st.sidebar.title("Hasil Prediksi")
predict = ''
if st.sidebar.button('Proses'):
    # Validate and convert input values
    validated_values = [
        validate_input(persentase_pm), validate_input(lama_sekolah),
        validate_input(pengeluaran_pk), validate_input(IPM),
        validate_input(UHH), validate_input(sanitasi_layak),
        validate_input(airminum_layak), validate_input(TPT),
        validate_input(TPAK), validate_input(PDRB)
    ]

    # Check if any value is None (indicating invalid input)
    if None in validated_values:
        st.sidebar.error("Masukkan nilai numerik yang valid untuk semua input.")
    else:
        predict = model.predict([validated_values])[0]
        prediction_label = "Tingkat Kemiskinan Rendah" if predict == 0 else "Tingkat Kemiskinan Tinggi"
        st.sidebar.success(f'Hasil Prediksi: {prediction_label}')

# Visualisasi Pohon Keputusan in the sidebar
dot_data = export_graphviz(model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
st.sidebar.graphviz_chart(graph)

# Main content (you can keep this part as it is)
st.write("Isi Utama Aplikasi Streamlit Anda di Sini.")
