## Laporan Proyek Machine Learning 

### Nama : Vika Fatnawati
### Nim : 211351148
### Kelas : Malam B

## Domain Proyek
dikarenakan tingkat kemiskinan yang ditinggi di beberapa wilayah di indonesia menjadi alasan penting mengapa proyek ini diangkat, dengan adanya sistem yang membantu dalam proses klasifikasi tingkat kemiskinan diharapkan agar membantu dalam pencegahan peningkatan tingkat kemiskinan di Indonesia. 

## Business Understanding

Berdasarkan penjelasan dari domain proyek diatas, maka perlu dilakukannya pembuatan sistem yang mampu mengklasifikasi tingkat kemiskinan di indonesia berdasarkan beberapa atribut yang di perlukan, pembuatan sistem ini menggunakan metode klasifikasi dengan algoritma decission tree.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Tingkat kemiskinan yang tinggi
- diperlukannya sistem yang mempermudah dalam proses klasifikasi

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- dapat mengurangi tingkat kemiskinan di beberapa wilayah di indonesia dikarenakan adanya sistem yang mampu mengklasifikasi tingkat kemiskinan apakah rendah atau tinggi sehingga dapat diambil sebuah langkah untuk menangani permasalahan tersebut. 

    ### Solution statements
    - Pembuatan sistem klasifikasi tingkat kemiskinan di beberapa wilayah di Indonesia dengan keluaran klasfikasi tingkat kemiskinan yang rendah atau tinggi menggunakan algoritma decision tree
    - metrik evaluasi yang digunakan adalah akurasi dan confusion matrix

## Data Understanding
Dataset yang dipakai diambil dari kaggle yang berisi 999 baris dengan 13 kolom yang diunggah oleh Ermila yang mana kebanyakan dari fiturnya memiliki tipe data object yang mana harus di transformasi menjadi numeric dikarenakan menggunakan "." bukan "," untuk angka desimalnya.

dataset: [Klasifikasi Tingkat Kemiskinan di Indonesia](https://www.kaggle.com/datasets/ermila/klasifikasi-tingkat-kemiskinan-di-indonesia).

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Provinsi = Provinsi yang ada di Indonesia
- persentase_pm = Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen) 
- lama_sekolah = Rata-rata Lama Sekolah Penduduk 15+ (Tahun)
- pengeluaran_pk = Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)
- IPM = Indeks Pembangunan Manusia
- UHH = Umur Harapan Hidup (Tahun)
- sanitasi_layak = Persentase rumah tangga yang memiliki akses terhadap sanitasi layak
- airminum_layak = Persentase rumah tangga yang memiliki akses terhadap air minum layak
- TPT = Tingkat Pengangguran Terbuka
- TPAK = Tingkat Partisipasi Angkatan Kerja
- PDRB = PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)

Didalam dataset tersebut juga terdapat data yang:
1. Null:

<img width="359" alt="image" src="https://github.com/vikaf26/uas-d3/assets/149370346/4b5e1a40-c138-4038-afad-f5961445aad2">

2. Duplikat:

<img width="115" alt="image" src="https://github.com/vikaf26/uas-d3/assets/149370346/6d63e0b9-3a34-4754-a7b0-bea377f05d3b">


## EDA
1. Klasifikasi Kemiskinan:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/e62c0fb5-e4ed-4264-bdce-c7b1d9ad6557)


didalam dataset terdapat sekitar 87% wilayah di indonesia dengan tingkat kemiskinan rendah dan 12% tinggi

2. Top 10 Provinsi dengan jumlah kabupaten terbanyak di indonesia
   
![image](https://github.com/vikaf26/uas-d3/assets/149370346/8cbba71e-9f2a-42bb-8408-72f6ca528252)

Jawa Timur menjadi provinsi yang memiliki kabupaten terbanyak di Indonesia diantara provinsi lainnya.

3. Top 10 Provinsi dengan tingkat kemiskinan tinggi:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/52c16dd5-3167-458e-b521-4bd31b0d0f9f)

Papua menjadi provinsi dengan tingkat kemiskinan tertinggi diantara provinsi lainnya.

4. Top 10 Provinsi dengan tingkat kemiskinan rendah:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/2b6f8480-0006-4f52-ab83-576144ad6f0c)

Jawa Timur menjadi provinsi yang memiliki tingkat kemiskinan ter-rendah diantara provinsi lainnya.

5. Tingkat pengangguran terbuka per Provinsi:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/efd99511-82a1-4642-b9bb-1d3af21f1c87)

Jawa Barat menjadi provinsi yang memiliki tingkat pengangguran terbuka yang paling tinggi diantara provinsi lainnya.

6. Tingkat partisipasi angkatan kerja per Provinsi:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/9d58ba85-9fd1-4946-9cb2-0d49fe05cd4e)

Jawa Timur adalah provinsi yang memiliki tingkat partisipasi angkatan kerja yang tinggi diantara provinsi lainnya.

## Data Preparation
1. menghapus data yang Null:
```
df.dropna(inplace=True)
df.isnull().sum()
```

2. menghapus fitur yang tidak akan dipakai:
```
df = df.drop(['Kab/Kota'],axis=1)
```
3. transformasi data object menjadi numeric:
```
# Melakukan label encoding
label_encoder = LabelEncoder()
df['Provinsi'] = label_encoder.fit_transform(df['Provinsi'])
```

## Modeling
1. menentukan nilai X dan Y:
```
x = df.drop(['Klasifikasi Kemiskinan'],axis=1)
y = df['Klasifikasi Kemiskinan']
```
2. pembentukan model prediksi dan cek akurasi:
```
# Inisialisasi model Decision Tree Classifier
dtc = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0,
    random_state=42, splitter='best'
)

# Latih model
model = dtc.fit(x_train, y_train)

# Prediksi label untuk data testing
y_pred = dtc.predict(x_test)

dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

print(f"akurasi data training = {accuracy_score(y_train, dtc.predict(x_train))}")
print(f"akurasi data testing = {dtc_acc} \n")
```
```
akurasi data training = 0.9944289693593314
akurasi data testing = 0.9354838709677419
````

## Evaluation
Setelah pembetukan model maka selanjutnya pengeekan hasil prediksi dengan mencoba model yang sudha di buat:

```
input_data = (0,18.98,9.48,7148.0,66.41,65.28,71.56,87.45,5.71,71.15,1648096.0)

input_data_as_numpy_array = np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('Klasifikasi Kemiskinan Rendah')
else:
  print('Klasifikasi Kemiskinan Tinggi')
```
```
[0.]
Klasifikasi Kemiskinan Rendah
```
keluaran yang ditampilkan sudah sesuai dengan data yang diuji. selanjutnya evaluasi model menggunakan confusion matrix:

```
# Menampilkan confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Visualisasi confusion matrix menggunakan heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=dtc.classes_, yticklabels=dtc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
```

![image](https://github.com/vikaf26/uas-d3/assets/149370346/8950909d-8302-4993-a2db-b1fda6945447)

dari total 155 data testing didapatkan hasil:
- 133 data benar di prediksi dengan hasil tingkat kemiskinan rendah
- 4 data salah di prediksi dengan hasil tingkat kemiskinan rendah
- 12 data benar di prediksi dengan hasil tingkat kemiskinan tingi
- 6 data salah di prediksi dengan hasil tingkat kemiskinan tinggi

## Visualisasi Hasil Pohon:

![image](https://github.com/vikaf26/uas-d3/assets/149370346/ad8957b6-390c-4f82-a143-fb076d3af01d)


## Deployment

[Link Aplikasi](https://uas-vika-d3.streamlit.app/)

<img width="960" alt="image" src="https://github.com/vikaf26/uas-d3/assets/149370346/8f03a777-b767-491b-983c-3c1c6716c9ce">
