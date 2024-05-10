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
analysis_choice = st.sidebar.selectbox("Select Analysis", ["Overview", "Correlation", "Distribution"])

if analysis_choice == "Overview":
    st.write("### Dataset Overview")
    st.write(df.head())

    st.write("### Dataset Description")
    st.write(df.describe())

elif analysis_choice == "Correlation":
    st.write("### Correlation Heatmap")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

elif analysis_choice == "Distribution":
    st.write("### Distribution of Features")
    feature_choice = st.selectbox("Select Feature", df.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature_choice], kde=True)
    plt.xlabel(feature_choice)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature_choice}")
    st.pyplot()

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