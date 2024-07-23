import pandas as pd
import numpy as np
import streamlit as st
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Definisi kelas LVQ
class LVQ:
    def __init__(self, n_prototypes=2, alpha=0.01, learning_rate=0.1, max_epochs=100, min_error=0.01):
        self.n_prototypes = n_prototypes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.min_error = min_error

    def inisialisasi_bobot(self, X, y):
        kelas_unik = np.unique(y)
        self.prototypes = []
        self.prototype_labels = []
        for kelas in kelas_unik:
            indeks_kelas = np.where(y == kelas)[0]
            indeks_dipilih = np.random.choice(indeks_kelas, self.n_prototypes, replace=False)
            for idx in indeks_dipilih:
                self.prototypes.append(X[idx])
                self.prototype_labels.append(kelas)
        self.prototypes = np.array(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)

    def perbarui_bobot(self, X, y):
        epoch = 0
        eps = 1

        while epoch < self.max_epochs or self.alpha > eps:
            epoch += 1
            for xi, yi in zip(X, y):
                # Menghitung jarak ke setiap prototype
                jarak = np.linalg.norm(self.prototypes - xi, axis=1)

                # Menemukan prototype terdekat
                indeks_pemenang = np.argmin(jarak)
                label_pemenang = self.prototype_labels[indeks_pemenang]

                # Memperbarui prototype
                if label_pemenang == yi:
                    self.prototypes[indeks_pemenang] += self.alpha * (xi - self.prototypes[indeks_pemenang])
                else:
                    self.prototypes[indeks_pemenang] -= self.alpha * (xi - self.prototypes[indeks_pemenang])

            # Mengurangi laju pembelajaran
            self.alpha -= self.alpha * self.learning_rate

    def fit(self, X, y):
        self.inisialisasi_bobot(X, y)
        self.perbarui_bobot(X, y)

    def predict(self, X):
        y_pred = []
        for xi in X:
            jarak = np.linalg.norm(self.prototypes - xi, axis=1)
            indeks_pemenang = np.argmin(jarak)
            y_pred.append(self.prototype_labels[indeks_pemenang])
        return np.array(y_pred)

    def nilai_bobot(self):
        return self.prototypes, self.prototype_labels

# Aplikasi Streamlit
st.title("Klasifikasi Diabetes Menggunakan LVQ")

# Pengunggah file
file_diunggah = st.file_uploader("Pilih file CSV", type="csv")
if file_diunggah is not None:
    df = pd.read_csv(file_diunggah)
    
    st.write("Dataset:")
    st.write(df.head())
    
    st.write("Deskripsi Dataset:")
    st.write(df.describe())
    
    st.write("Info Dataset:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Praproses data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X[outliers] = np.where(X[outliers] < lower_bound, lower_bound, X[outliers])
    X[outliers] = np.where(X[outliers] > upper_bound, upper_bound, X[outliers])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pastikan setiap kelas memiliki minimal 20 data
    kelas_0 = df[df.iloc[:, -1] == 0]
    kelas_1 = df[df.iloc[:, -1] == 1]

    if len(kelas_0) >= 20 and len(kelas_1) >= 20:
        kelas_0_sample = kelas_0.sample(n=20, random_state=42)
        kelas_1_sample = kelas_1.sample(n=20, random_state=42)
        
        data_sample = pd.concat([kelas_0_sample, kelas_1_sample])
        
        # Pisahkan fitur dan label dari data sample
        X_sample = data_sample.iloc[:, :-1]
        y_sample = data_sample.iloc[:, -1]

        # Encode label
        y_sample_encoded = le.fit_transform(y_sample)

        # Normalisasi fitur
        X_sample_scaled = scaler.fit_transform(X_sample)

        # Bagi data sample menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X_sample_scaled, y_sample_encoded, test_size=0.2, random_state=42)

        st.write(f"Ukuran data train: {X_train.shape[0]}")
        st.write(f"Ukuran data uji: {X_test.shape[0]}")

        # Inisialisasi dan latih model LVQ
        lvq = LVQ(n_prototypes=4, alpha=0.01, learning_rate=0.2, max_epochs=150, min_error=0.01)
        lvq.fit(X_train, y_train)
        
        y_test_pred = lvq.predict(X_test)
        test_accuracy = (y_test_pred == y_test).mean()
        st.write(f"Akurasi Uji: {test_accuracy:.2f}")

        bobot, bobot_target = lvq.nilai_bobot()
        st.write("Bobot Akhir:")
        for proto, label in zip(bobot, bobot_target):
            st.write(f"Kelas {label}: {proto}")

        st.write("Target Aktual vs Target Prediksi :")
        perbandingan = pd.DataFrame({"Aktual": y_test, "Prediksi": y_test_pred})
        st.write(perbandingan)

        # Input prediksi
        st.write("## Prediksi Diabetes")
        input_data = []
        for col in X.columns:
            nilai = st.number_input(f"Masukkan {col}", value=0.0)
            input_data.append(nilai)
        
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        
        if st.button("Prediksi"):
            prediksi = lvq.predict(input_data_scaled)
            prediksi_label = le.inverse_transform(prediksi)
            if prediksi_label[0] == 1:
                st.write("Kelas prediksi untuk data input adalah: 1 (Diabetes)")
            else:
                st.write("Kelas prediksi untuk data input adalah: 0 (Non-Diabetes)")
    else:
        st.write("Dataset tidak memiliki minimal 20 data untuk setiap kelas.")