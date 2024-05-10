import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

url = "Data_Cleaned.csv"
df = pd.read_csv(url)

st.title("Prediksi Harga Jual Mobil Bekas untuk Mendukung Infrastruktur Transportasi Berkelanjutan")
st.subheader("Dataset")
st.write(df.head(5))

st.subheader('Exploratory Data Analysis (EDA)')

st.title('Number of Cars by Seller Type')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Seller_Type', data=df, palette='pastel')
ax.set_xlabel('Seller Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Number of Cars by Seller Type', fontsize=14)
st.pyplot(fig)
st.write('Pada visualisasi di atas adalah untuk melihat perbandingan jumlah mobil bekas yang dijual "Individu / 1" dan "Dealer / 0". Berdasarkan plot bar yang terlihat "Dealer" menjual lebih banyak mobil bekas daripada "Individu". Jadi kita dapat mengambil sedikit kesimpulan bahwa distribusi mobil bekas ini lebih banyak dilakukan oleh "Dealer".')

st.title('Number of Cars by Manufacturing Year')
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='Year', data=df, palette='pastel', order=df['Year'].value_counts().index)
ax.set_xlabel('Manufacturing Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Number of Cars by Manufacturing Year', fontsize=14)
plt.xticks(rotation=45) 
st.pyplot(fig)
st.write('Pada visualisasi selanjutnya disini menggambarkan distribusi jumlah mobil bekas berdasarkan tahun pembuatannya. Disini kita bisa melihat tren ataupun adanya peningkatan maupun penurunan dalam jumlah mobil yang di produksi dari tahun ke tahun.')

st.title('Comparison of Selling Price by Transmission Type')
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Transmission', y='Selling_Price', data=df, palette='pastel')
ax.set_xlabel('Transmission Type', fontsize=12)
ax.set_ylabel('Selling Price', fontsize=12)
ax.set_title('Comparison of Selling Price by Transmission Type', fontsize=14)
st.pyplot(fig)
st.write('Selanjutnya pada visualisasi Boxplot di atas merupakan perbandingan distribusi harga jual mobil bekas berdasarkan tipe atau jenis transmisinya, yaitu : transmisi manual dan otomatis. Dari visualisasi tersebut dapat sedikit diambil kesimpulan bahwa mobil bekas dengan transmisi otomatis cenderung memiliki harga jual yang lebih tinggi daripada mobil bekas dengan transmisi manual.')

file_path = 'dtc_model.pkl'

# Input data fields
Year = st.number_input('Year', value=0)
Present_Price = st.number_input('Present Price', value=0.0)
Kms_Driven = st.number_input('Kms Driven', value=0)
Fuel_Type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
Seller_Type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
Transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
Owner = st.number_input('Owner', value=0)

# Encoding categorical variables
Fuel_Type_enc = 1 if Fuel_Type == 'Diesel' else 2 if Fuel_Type == 'CNG' else 0
Seller_Type_enc = 1 if Seller_Type == 'Individual' else 0
Transmission_enc = 1 if Transmission == 'Automatic' else 0

if st.button('Predict'):
    # Load the model
    try:
        with open(file_path, 'rb') as file:
            clf = joblib.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    
    # Check if model loaded successfully
    if 'clf' in locals():
        # Perform prediction
        input_data = [[Year, Present_Price, Kms_Driven, Fuel_Type_enc, Seller_Type_enc, Transmission_enc, Owner]]
        result = clf.predict(input_data)

        if result.size > 0:
            st.write('Predicted Selling Price:', result[0])
        else:
            st.write('Sorry, unable to make prediction. Please check your input.')
