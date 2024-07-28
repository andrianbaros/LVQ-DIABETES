import streamlit as st
import pandas as pd
import numpy as np
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

def initialize_prototypes(df):
    # Menggunakan baris pertama dan kedua sebagai prototipe
    X = df.iloc[:, :-1].values
    return X[:2], np.unique(df.iloc[:, -1].values)

# Fungsi pelatihan LVQ
def train_lvq(df, prototypes, classes, n_epochs):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    epoch_data = []
    
    for epoch in range(n_epochs):
        # Menentukan learning rate untuk epoch saat ini
        if epoch == 0:
            learning_rate = 0.5
        elif epoch == 1:
            learning_rate = 0.05
        else:
            learning_rate = np.random.uniform(0.01, 1.0)

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
        epoch_data.append((epoch + 1, learning_rate, prototypes.copy()))
    return prototypes, epoch_data

# Fungsi untuk menghitung akurasi
def calculate_accuracy(df, prototypes, classes):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
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

# Initialize prototypes directly from df.head(2)
initial_prototypes, classes = initialize_prototypes(df)
print("Initial Prototypes:\n", initial_prototypes)

# Antarmuka pengguna Streamlit
st.sidebar.header('Parameter Input Pengguna')
n_epochs = st.sidebar.slider('Jumlah Epoch', 10, 150, 150)

st.subheader('Lakukan Training Terlebih Dahulu Dengan Mengatur Jumlah Epoch Lalu Latih Model!')
if st.button('Latih Model'):
    prototypes = initialize_prototypes(df)[0]
    
    st.subheader('Bobot Awal')
    st.write(pd.DataFrame(prototypes, columns=df.columns[:-1]))
    
    initial_prototypes = prototypes.copy()
    
    prototypes, epoch_data = train_lvq(df, prototypes, classes, n_epochs)
    
    st.text('Pelatihan model selesai!')

    accuracy = calculate_accuracy(df, prototypes, classes)
    st.subheader('Akurasi Model')
    st.write(f'Akurasi: {accuracy * 100:.2f}%')

    st.subheader('Detail Algoritma')
    st.write(f'Jumlah Epoch: {n_epochs}')
    st.write(f'Kelas: {classes}')
    
    st.subheader('Bobot Akhir')
    st.write(pd.DataFrame(prototypes, columns=df.columns[:-1]))
    st.write(f'Kelas: {classes}')
    
    st.subheader('Detail Epoch')
    # Membuat DataFrame untuk menampilkan detail epoch
    epoch_details = []
    for epoch, lr, proto in epoch_data:
        for i, p in enumerate(proto[:2]):  # Menampilkan 2 data untuk 1 epoch
            # Menghitung bobot akhir (jarak) untuk kelas 0 dan kelas 1 menggunakan rumus
            bobot_kelas_0 = np.sqrt(np.sum((p - initial_prototypes[0]) ** 2))
            bobot_kelas_1 = np.sqrt(np.sum((p - initial_prototypes[1]) ** 2))
            epoch_dict = {
                'Epoch': epoch,
                'Learning Rate': lr,
                'Bobot Kelas 0': bobot_kelas_0,
                'Bobot Kelas 1': bobot_kelas_1,
            }
            epoch_dict.update({f'W{j+1}': feature for j, feature in enumerate(p)})
            epoch_details.append(epoch_dict)
        
    epoch_df = pd.DataFrame(epoch_details)
    
    st.write(epoch_df)

    # Menyimpan model yang telah dilatih dan scaler di session state
    st.session_state['prototypes'] = prototypes
    st.session_state['classes'] = classes

# Input dengan nama deskriptif
st.subheader('Prediksi Diabetes')
input_features = {}
feature_names = df.columns[:-1]

for feature in feature_names:
    input_features[feature] = st.number_input(f'{feature}', value=0.0)

if st.button('Prediksi'):
    if 'prototypes' in st.session_state and 'classes' in st.session_state:
        input_sample = np.array([input_features[feature] for feature in feature_names]).reshape(1, -1)
        prediction = predict(input_sample[0], st.session_state['prototypes'], st.session_state['classes'])
        if prediction == 0:
            st.write("Prediksi: Non-diabetes (0)")
        else:
            st.write("Prediksi: Diabetes (1)")
    else:
        st.write("Silakan latih model terlebih dahulu.")

st.sidebar.markdown("[Lihat Source Code](https://github.com/andrianbaros/LVQ-DIABETES)")
