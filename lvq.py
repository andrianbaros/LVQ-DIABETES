import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import io

# Aplikasi Streamlit
st.title("Klasifikasi Diabetes Menggunakan LVQ")
st.title("KELOMPOK 7")
st.write("Anggota :")
st.write("10122003 - Andrian Baros")
st.write("10122017 - M. Fathi Zaidan")
st.write("10122036 - Khotibul Umam")
st.write("10122506 - Arya Ababil")

# Membaca file CSV
file_path = "diabetes(LVQ, KNN, KMEANS).csv"
df = pd.read_csv(file_path)

st.write("Dataset:")
st.write(df.head(2))

st.write("Deskripsi Dataset:")
st.write(df.describe())

st.write("Info Dataset:")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Preprocessing data
def preprocess_data(data, scaler=None):
    data = data.dropna()  # Menghapus nilai yang hilang
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler

# Inisialisasi prototipe
def initialize_prototypes(X, y):
    classes = np.unique(y)
    prototypes = []
    for c in classes:
        class_samples = X[y == c]
        prototype = random.choice(class_samples)
        prototypes.append(prototype)
    return np.array(prototypes), classes

# Fungsi pelatihan LVQ
def train_lvq(X, y, prototypes, classes, learning_rate, n_epochs):
    epoch_data = []
    for epoch in range(n_epochs):
        for i in range(len(X)):
            sample = X[i]
            label = y[i]
            distances = np.linalg.norm(prototypes - sample, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_prototype = prototypes[nearest_idx]
            if classes[nearest_idx] == label:
                prototypes[nearest_idx] += learning_rate * (sample - nearest_prototype)
            else:
                prototypes[nearest_idx] -= learning_rate * (sample - nearest_prototype)
        
        # Menyimpan posisi prototipe untuk epoch saat ini
        epoch_data.append((epoch + 1, prototypes.copy()))
    return prototypes, epoch_data

# Fungsi untuk menghitung akurasi
def calculate_accuracy(X, y, prototypes, classes):
    correct_predictions = 0
    for i in range(len(X)):
        sample = X[i]
        label = y[i]
        distances = np.linalg.norm(prototypes - sample, axis=1)
        nearest_idx = np.argmin(distances)
        if classes[nearest_idx] == label:
            correct_predictions += 1
    accuracy = correct_predictions / len(X)
    return accuracy

# Fungsi untuk membuat prediksi
def predict(sample, prototypes, classes):
    distances = np.linalg.norm(prototypes - sample, axis=1)
    nearest_idx = np.argmin(distances)
    return classes[nearest_idx]

# Preprocessing data
X, y, scaler = preprocess_data(df)

# Antarmuka pengguna Streamlit
st.sidebar.header('Parameter Input Pengguna')
learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
n_epochs = st.sidebar.slider('Jumlah Epoch', 10, 1000, 100)

st.subheader('Lakukan Training Terlebih Dahulu Dengan Mengatur Learning Rate dan Jumlah Epoch Lalu Latih Model!')
if st.button('Latih Model'):
    prototypes, classes = initialize_prototypes(X, y)
    
    st.subheader('Bobot Awal')
    st.write(prototypes)
    
    initial_prototypes = prototypes.copy()
    
    prototypes, epoch_data = train_lvq(X, y, prototypes, classes, learning_rate, n_epochs)
    
    st.text('Pelatihan model selesai!')

    
    accuracy = calculate_accuracy(X, y, prototypes, classes)
    st.subheader('Akurasi Model')
    st.write(f'Akurasi: {accuracy * 100:.2f}%')

    st.subheader('Detail Algoritma')
    st.write(f'Learning Rate: {learning_rate}')
    st.write(f'Jumlah Epoch: {n_epochs}')
    st.write(f'Kelas: {classes}')
    
    
    st.subheader('Bobot Akhir')
    st.write(prototypes)
    st.write(f'Kelas: {classes}')
    
    st.subheader('Detail Epoch')
    # Membuat DataFrame untuk menampilkan detail epoch
    epoch_details = []
    for epoch, proto in epoch_data:
        epoch_dict = {'Epoch': epoch}
        for i, p in enumerate(proto):
            if i == 0:
                epoch_dict.update({f'W0{j+1}': feature for j, feature in enumerate(p)})
            elif i == 1:
                epoch_dict.update({f'W1{j+1}': feature for j, feature in enumerate(p)})
        epoch_details.append(epoch_dict)
        
    epoch_df = pd.DataFrame(epoch_details)
    
    st.write(epoch_df)

    # Menyimpan model yang telah dilatih dan scaler di session state
    st.session_state['prototypes'] = prototypes
    st.session_state['classes'] = classes
    st.session_state['scaler'] = scaler

# Input dengan nama deskriptif
st.subheader('Prediksi Diabetes')
input_features = {}
feature_names = df.columns[:-1]

for feature in feature_names:
    input_features[feature] = st.number_input(f'{feature}', value=0.0)

if st.button('Prediksi'):
    if 'prototypes' in st.session_state and 'classes' in st.session_state and 'scaler' in st.session_state:
        input_sample = np.array([input_features[feature] for feature in feature_names]).reshape(1, -1)
        input_sample_scaled = st.session_state['scaler'].transform(input_sample)
        prediction = predict(input_sample_scaled[0], st.session_state['prototypes'], st.session_state['classes'])
        if prediction == 0:
            st.write("Prediksi: Non-diabetes (0)")
        else:
            st.write("Prediksi: Diabetes (1)")
    else:
        st.write("Silakan latih model terlebih dahulu.")

st.sidebar.markdown("[Lihat Source Code](https://github.com/andrianbaros/LVQ-DIABETES)")
