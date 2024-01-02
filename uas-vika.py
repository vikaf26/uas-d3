import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier

model = pickle.load(open('d3_tingkat_kemiskinan.sav', 'rb'))

st.title('Klasifikasi Tingkat Kemiskinan')

#list provinsi
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

col1,col2 = st.columns(2)

with col1:
    Provinsi = st.selectbox("Pilih Provinsi:", provinsi_list)
    persentase_pm = st.text_input('Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)')
    lama_sekolah = st.text_input("Rata-rata Lama Sekolah Penduduk 15+ (Tahun)")
    pengeluaran_pk = st.text_input("Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)")
    IPM = st.text_input("Indeks Pembangunan Manusia")
    UHH = st.text_input("Umur Harapan Hidup (Tahun)")
with col2:
    sanitasi_layak = st.text_input("Persentase rumah tangga yang memiliki akses terhadap sanitasi layak")
    airminum_layak = st.text_input("Persentase rumah tangga yang memiliki akses terhadap air minum layak")
    TPT = st.text_input("Tingkat Pengangguran Terbuka")
    TPAK = st.text_input("Tingkat Partisipasi Angkatan Kerja")
    PDRB = st.text_input("PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)")

encoder_prov = LabelEncoder()
encoder_prov.fit(provinsi_list)
prov_encoded = encoder_prov.transform([Provinsi])[0]

predict = ''

if st.button('Estimasi Berat Ikan'):
    predict = model.predict(
        [[prov_encoded,persentase_pm,lama_sekolah,pengeluaran_pk,
          IPM,UHH,sanitasi_layak,airminum_layak,TPT,TPAK,PDRB]]
    )
    st.write ('Klasifikasi Tingkat Kemiskinan : ', predict)

# Visualisasi Pohon Keputusan
    dot_data = export_graphviz(model, out_file=None, feature_names=model.feature_names_,
                               class_names=model.classes_, filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)