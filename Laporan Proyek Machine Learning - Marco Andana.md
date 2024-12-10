# Laporan Proyek Machine Learning - Marco Andana

## Domain Proyek

Netflix adalah sebuah perusahaan yang bergerak di bidang streaming dan mencakup pasar streaming global. Perusahaan ini mengalami pertumbuhan pesat beberapa tahun ini. Pergerakan harga saham perusahaan netflix dipengaruhi oleh beberapa faktor seperti laporan keungan, perilisan film baru, pertumbuhan jumlah pelanggan, kinerja perusahaan, dan lainnya. Namun, Harga saham yang fluktuatif sering kali menjadi tantangan tersendiri bagi para investor dan analis.

- Prediksi harga saham merupakan aspek penting dalam pengambilan keputusan investasi. Dengan analisis dan prediksi harga saham yang akurat, investor dapat mengambil keputusan yang lebih baik terkait pembelian ataupun penjualan saham. Dalam konteks ini, analisis deret waktu (Time Series) menjadi alat yang relevan dan penting. Dengan menggunakan data historis pergerakan harga saham netflix, model prediktif berbasis time series seperti KNN, Random Forest, dan Boosting Algorithm menjadi alat yang dapat membantu memprediksi pergerakan harga saham netflix di masa depan sehingga keputusan yang lebih baik dapat di ambil.
  
   Referensi: 
    [Apa itu Netflix?](https://help.netflix.com/id/node/412) 
    [6 Faktor Mempengaruhi Harga Saham](https://pluang.com/blog/academy/equity-101/6-faktor-mempengaruhi-harga-saham)
    [Investor Pemula Harus Tahu: Analisa Saham Secara Fundamental ](https://www.mncsekuritas.id/pages/investor-pemula-harus-tahu-analisa-saham-secara-fundamental/)


## Business Understanding

Sebagai seorang investor saham, tentu kita mencari keuntungan dari membeli saham. Harga saham yang fluktuatif menjadi pertimbangan bagi para investor dalam melakukan penjualan atau pembelian saham. Beberapa faktor seperti tren pasar, kinerja perusahaan, dan faktor eksternal lainnya menjadi pengaruh naik atau turunnya harga saham. Dengan menggunakan alat prediksi harga saham ini, Investor dan analis dapat terbantu dan mendapatkan pandangan terbaik terkait keputusan yang harus diambil untuk kedepannya.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh untuk saham Netflix ?
- Bagaimana cara untuk memprediksi harga saham netflix secara akurat berdasarkan data historis ?
- Bagaimana mengidentifikasi tren harga saham netflix di masa depan ?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan analisis korelasi untuk mengidentifikasi fitur-fitur yang ada seperti harga pembukaan (Open), harga tertinggi (High), harga terendah (Low), harga penutupan (Close), volume perdagangan (Volume), dan moving average (MA).
- Mengembangkan tiga algoritma seperti K-Nearest Neighbors (KNN), Random FOrest (RF), dan Boosting lalu melakukan evaluasi dengan metrik seperti MAE untuk mencari model dengan akurasi terbaik.
- Menggunakan data historis dan indikator teknis seperti moving avarage (MA7, MA30) dan volatilitas untuk mendeteksi pola dalam tren harga saham.

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengembangkan model tiga prediksi seperti K-Nearest Neighbors (KNN), Random Forest (RF), dan Boosting lalu membandingkan ketiga model tersebut dan memvisualisasikan tingkat error yang di dapatkan melalui pengujian menggunakan metrik seperti Mean Absolute Error (MAE) pada train dan test dataset untuk mencari mana model terbaik yang dapat digunakan untuk melakukan prediksi harga saham.
    - Membuat prediksi jangka pendek dan menengah menggunakan moving avarage (MA7, MA30) untuk membantu investor dalam memprediksi dan menganalisis pola pergerakan harga saham dengan lebih baik dan akurat.

## Data Understanding
Dataset yang saya gunakan pada submission kali ini adalah Netflix Stock Price Prediction. Dataset ini memiliki 7 karakteristik atau fitur utama seperti Date, Open, High, Low, Close, Adj. Close, dan Volume. Dataset ini memiliki total 1009 kolom data. Data ini adalah data yang diambil dengan rentang waktu 4 tahun, dimulai dari 5 Febuari 2018 sampai 5 Febuari 2022. Dari dataset ini, tidak ditemukan adanya outlier, atau duplicate pada data. Namun terdapat missing value sehingga harus diproses dengan cara penghapusan.

Sumber Data diambil dari Kaggle [Netflix Stock Price Prediction](https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction/data).

### Variabel-variabel pada Netflix Stock Price Prediction dataset adalah sebagai berikut:
- Date : merupakan tanggal lengkap dengan format [yyyy-mm-dd] terkait informasi saham.
- Open: Harga pembukaan saham pada hari tertentu.
- High: Harga tertinggi saham pada hari tertentu.
- Low: Harga terendah saham pada hari tertentu.
- Close: Harga penutupan saham pada hari tertentu.
- Volume: Jumlah saham yang diperdagangkan.
 ##### Variabel turunan
- Moving average (MA): rata-rata pergerakan harga saham selama 7 hari dan 30 hari terakhir.
- volatilitas: menggambarkan fluktuasi harga saham.
- lagged prices: harga penutupan saham sehari sebelumnya dan seminggu sebelumnya.

![Netflix stock price plot](https://github.com/user-attachments/assets/2e1917c0-cf0b-41aa-8cc4-6f7aa2d82725)


Gambar diatas merupakan visualisasi dari harga saham netflix dari tahun 2018-2022

![Explorenatory Data Analysis - Univariate Analysis](https://github.com/user-attachments/assets/b702799a-a4e4-4b6c-805c-d216e94bab25)


Gambar diatas merupakan visualisasi menggunakan histogram dan berdasarkan histogram diatas, dapat ditarik kesimpulan sebagai berikut:
- Dilihat dari histogram diatas, fitur utama (Open, High, Low, Close, Adj. Close) memiliki sebagian besar nilai pada range 300-600 yang menandakan bahwa harga saham netflix cenderung berada pada rentang nilai tersebut.
- Distribusi model tidak normal atau miring ke kanan (right skewed)

## Data Preparation
Pada bagian data preparation ini, dilakukan penambahan fitur. Tujuannya adalah untuk membantu model dalam lebih memahami data.

## 1.Lag features (Fitur Historis)
#### Menambahkan harga saham dari beberapa hari sebelumnya
```
df['Close_Lag1'] = df['Close'].shift(1)  # Harga penutupan 1 hari sebelumnya
df['Close_Lag7'] = df['Close'].shift(7)  # Harga penutupan 7 hari sebelumnya
```

## 2. Moving Averages
### Tambahkan rata-rata bergerak untuk menangkap tren jangka pendek:
```
df['MA7'] = df['Close'].rolling(window=7).mean()  # Rata-rata 7 hari
df['MA30'] = df['Close'].rolling(window=30).mean()  # Rata-rata 30 hari
```

## 3. Fitur Tanggal
### Ekstrak informasi dari tanggal, seperti hari dalam seminggu atau bulan:
```
# 0 = Senin, 6 = Minggu
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday
```

## 4. Volatilitas (Spread Harian)
### Tambahkan fitur volatilitas:
```
df['Volatility'] = df['High'] - df['Low']
```

## 5. Menghapus Nilai NAN
### Menghapus nilai NAN
```
df.dropna(inplace=True)
```

## 6. Train test split
Melakukan pembagian dataset menjadi train dataset dan test dataset dengan rasio 80:20. Pembagian ini sudah ideal mengingat dataset yang dimiliki hanya 1009 yang masuk dalam kategori sedikit.

- Data preparation yang dilakukan adalah dengan menambahkan beberapa fitur tambahan agar model dapat lebih mudah memahami data yang digunakan untuk pelatihan. Penambahan seperti moving avarages, lag features, fitur tanggal dan volatilitas dapat membantu model agar bisa memahami tren pasar yang terjadi serta membuat analisis lebih akurat karena terdapat informasi rata-rata yang bisa menangkap tren jangka pendek. Pembagian dataset menjadi train dan test dataset dengan rasio 80:20 juga merupakan hal yang tidak kalah penting dilakukan pada saat data preparation. Hal ini untuk mencegah terjadinya **Data Leak** yang menyebabkan model memiliki informasi terkait distribusi data test dan data uji apabila akan dilakukan scaling data seperti normalisasi dan standarisasi.
- Tahap data preparation ini diperlukan agar model dapat mengkap tren jangka pendek dan lebih memahami data latihan agar dapat meningkatkan akurasi dan ketepatan model serta pembagian 80:20 adalah jumlah yang ideal karena dataset hanya terdiri dari 1009 data yang mana pembagian 80% train dan 20% test sudah cocok untuk dataset yang berukuran kecil.

## Modeling
Pada tahap modeling, proyek ini menggunakan 3 model untuk diuji coba, yaitu K-Nearest Neighbors (KNN), Random Forest (RF), dan Boosting. 

### 1. K-Nearest Neighbors (KNN)
Model KNN ini adalah model sederhana yang menggunakan 'Kesamaan fitur' untuk memprediksi nilai dari setiap data yang baru. Model ini mencari titik K terdekat dari titik X (titik awal) dengan memperhitungkan jarak. Setiap titik X akan mencari titik K sejumlah dengan pemilihan jumlah K yang ditentukan. Misalnya apabila K = 1 maka model akan mencari titik K terdekat sebanyak 1, begitu juga apbila K = 10 maka model akan mencari 10 titik K terdekat dari titik X. Pemilihan nilai K sangat penting karena apabila nilai k yang dipilih terlalu kecil maka akan menghasilkan model yang overfit dan varians yang tinggi, sedangkan apabila nilai k yang dipilih terlalu tinggi maka akan menghasilkan model yang underfit dan bias yang tinggi. Untuk menemukan nilai K yang tepat dapat dilakukan dengan mencoba beberapa nilai K, misalnya dari 1-20 kemudian membandingkan mana yang sesuai.

**Parameter**: N-neighbors 
ini merupakan parameter yang menentukan jumlah K. Pada proyek ini nilai K yang digunakan adalah 10.

#### Kelebihan:
1. Algoritma ini tidak memerlukan banyak asumsi atau parameter sehingga cocok digunakan pada dataset yang kecil dan tidak kompleks.
2. Adaptif terhadap data baru sehingga tidak perlu dilakukan pelatihan ulang bila ada data baru.
3. Efektif dalam menangani data non-linear.

#### Kekurangan:
1. Algoritma ini menjadi lambat jika dataset sangat besar karena harus menghitung jarak setiap kali ada prediksi.
2. Sensitif terhadap fitur yang tidak dinormalisasi seperti bila ada fitur dengan satuan kilometer dan satuan gram yang akan membuat hasil bias.
3. Curse of dimensionality yang dimana algoritma ini kesulitan menghadapi data dengan dimensi tinggi (banyak fitur).

### 2. Random Forest (RF)
Model Random Forest adalah model yang termasuk kedalam algoritma supervised learning. Model ini juga termasuk kedalam kategori ensembled (group) training. *Apa itu ensembled training ?* ensembled training adalah sebuah model prediksi yang terdiri dari beberapa model yang bekerjasama untuk menyelesaikan masalah secara bersama-sama. Pada ensembled training, setiap model membuat prediksi secara independen tanpa mempengaruhi satu sama lain. Hasil keluaran/prediksi dari masing-masing model ini kemudian digabungkan dan menjadi hasil akhir untuk prediksi.Ada dua teknik pendekatan pada ensembled training, yaitu Bagging dan Boosting. Random forest menggunakan pendekatan Bagging dimana setiap model dilatih dengan *sampling with replacement* yang mana setiap model menggunakan sampling yang berbeda-beda. Sampling yang berbeda ini tidak akan mempengaruhi satu sama lain sehingga setiap model yang bekerja akan berfungsi secara independen.Random forest pada dasarnya adalah kumpulan dari decision tree yang dimasukkan kedalam teknik bagging, dengan kata lain ada banyak sekali decision tree yang bekerja sama membentuk sebuah model yang disebut random forest. Hasil akhir dari model random forest untuk kasus klasifikasi adalah prediksi terbanyak dari seluruh pohon, sedangkan untuk kasus regresi adalah nilai rata-rata dari seluruh pohon.

**Parameter**: 
- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Proyek ini menggunakan angka 16.
- random_state: digunakan untuk mengontrol random number generator yang digunakan. Proyek ini menggunakan angka 55.
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel. Proyek ini menggunakan angka -1.

#### Kelebihan:
1. Resiko overfitting menjadi berkurang karena menggabungkan banyak decision tree.
2. Pararelisasi yang mudah karena setiap decision tree dilatih secara independen.
3. Tahan terhadap noise dan outlier karena RF mengambil rata-rata prediksi dari decision tree.

#### Kekurangan:
1. Kompleks dan lambat karena model yang besar dengan banyak decision tree membutuhkan waktu untuk pelatihan.
2. Kurang baik pada data dengan dimensi tinggi karena banyak decision tree mungkin memilih fitur yang kurang relevan.
3. Memori yang diperlukan tinggi karena harus membuat banyak decision tree.

### Boosting Algorithm
Boosting algorithm adalah salah satu dari dua pendekatan ensembled training. Boosting algorithm adalah model yang berlatih dengan memperbaiki kesalahan yang dilakukan oleh model sebelumnya. Pada tahap training, algoritma ini membuat model untuk dilatih dengan data yang ada, lalu algoritma ini akan membuat model baru untuk memperbaiki kesalahan dari model sebelumnya. Dengan kata lain algoritma ini terus menerus membuat model baru untuk memperbaiki kesalahan dari model sebelumnya sehingga menciptakan sebuah model yang memiliki akurasi tertinggi. Algoritma ini bekerja diawali dengan memberikan bobot yang sama pada tiap data, setiap kali ada data yang salah di prediksi maka data tersebut akan di berikan bobot yang tinggi, sedangkan data yang benar di prediksi akan di berikan bobot yang rendah. Model akan fokus menyelesaikan data dengan bobot tinggi. Proses iteratif ini berlanjut sampai model mendapatkan akurasi yang diinginkan.

**Parameter**:
- learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting. Proyek ini menggunakan angka 0.05.
- random_state: digunakan untuk mengontrol random number generator yang digunakan. Proyek ini menggunakan angka 55.

#### Kelebihan:
1. Kinerja yang baik karena teknik boosting sering memberikan akurasi yang baik dalam beberapa kompetisi machine learning.
2. Menangani bias dan varians dengan baik karena boosting mengoreksi kesalahan dari model sebelumnya yang menghasilkan model yang lebih baik dari waktu kewaktu.
3. Memprioritaskan kesalahan yang sulit. Algoritma boosting memberikan bobot tinggi pada data yang sulit diprediksi sehingga meningkatkan akurasi.

#### Kekurangan:
1. Overfitting jika tidak dikendalikan. Apabila parameter seperti learning rate atau jumlah estimator tidak disetting dengan baik maka dapat menyebabkan overfitting pada data pelatihan.
2. Lambat untuk dataset besar karena algoritma ini dilatih secara berurutan.
3. Sensitif terhadap noise karena model berfokus pada kesalahan dari model sebelumnya maka data yang mengandung banyak noise dapat mengganggu proses pelatihan.

- Dari ketiga algoritma diatas, KNN memiliki tingkat akurasi tertinggi karena tingkat error yang rendah yaitu hanya 1,4% saja saat diuji coba prediksi dan tingkat error 66.41 dengan metriks Mean Absolute Error(MAE).


## Evaluation
Pada tahap evaluasi model, KNN menjadi model yang paling mendekati nilai sebenarnya setelah dilakukan uji coba. Metriks yang digunakan untuk model ini adalah Mean Absolute Error (MAE). Berdasarkan evaluasi dari metrik MAE, dapat disimpulkan bahwa model KNN adalah model yang paling cocok dengan dataset ini karena memiliki error yang paling kecil.

![plot dengan bar chart](https://github.com/user-attachments/assets/6978f925-9f46-4740-a7c5-6412b3040efd)

Gambar plot dengan bar chart dari hasil train dan test.

![hasil prediksi](https://github.com/user-attachments/assets/7a9d0bb7-0274-47b3-82b9-40d0922f0d7c)


Gambar prediksi dengan perbandingan antara nilai sebenarnya dan prediksi masing-masing model.

![Gambar evaluasi error dengan Mean Absolute Error(MAE)](https://github.com/user-attachments/assets/f5bd3613-a67b-4461-9c72-e55aee272a24)

Gambar evaluasi error dengan MAE

Berdasarkan gambar diatas, Nilai MAE yang didapat adalah:
untuk KNN: 66.41
untuk RF: 306.66
untuk Boosting: 280.35

## Mean Absolute Error
Mean Absolute Error (MAE) adalah metrik yang mengukur seberapa dekat nilai prediksi model dengan nilai yang sebenarnya pada tugas regresi. MAE menghitung rata-rata selisih absolut antara nilai prediksi dengan nilai sebenarnya.

### Formula MAE
![Formula MAE](https://github.com/user-attachments/assets/b887f5e3-dbe0-4a4c-b5f4-b89e90366f14)

### Cara kerja MAE

#### 1. Hitung Selisih Absolut:
- Untuk setiap data, hitung selisih absolut antara nilai prediksi (^yi) dan nilai sebenarnya (yi).
 
#### 2. Rata-Rata Semua Selisih:
- Semua selisih absolut tersebut dijumlahkan, kemudian dibagi dengan jumlah total data (n).

#### 3. Hasil Akhir (MAE):
- Hasil akhirnya adalah nilai rata-rata dari semua kesalahan absolut. Nilai MAE selalu positif.

**Kesimpulan**
Berdasarkan proyek yang telah dilakukan, semua problem statements yang disebutkan di bagian atas telah terselesaikan. Mulai dari identifikasi fitur yang paling berpengaruh untuk penentuan harga saham, lalu cara untuk memprediksi harga saham secara akurat menggunakan model-model yang di jalankan dan di analisis, serta mengidentifikasi tren harga saham netflix di masa yang akan datang menggunakan data historis. Solution statement yang dijabarkan juga memberikan dampak dalam menjawab pertanyaan-pertanyaan yang telah dijabarkan pada problem statements sehingga goals-goals yang diharapkan sebelumnya telah terpenuhi dan tercapai.

**---Ini adalah bagian akhir laporan---**