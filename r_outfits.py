import pandas as pd
import numpy as np
import urllib.request
import requests
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

all = pd.read_csv('Data_Final.csv')
# Fungsi untuk mengonversi suhu dari Kelvin ke Celcius
def kelvin_to_celcius(kelvin):
    celcius = kelvin - 273.15
    return celcius

# Fungsi untuk mengonversi suhu ke kategori
def convert_temperature(temperature):
    if temperature < 20:
        temperature = "Cold"
    elif temperature >= 20 and temperature < 30:
        temperature = "Normal"
    elif temperature >= 30:
        temperature = "Hot"
    return temperature

# Fungsi utama untuk merekomendasikan pakaian
def recommend_outfits(gender_product, city):
    # Mendapatkan data cuaca
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    API_KEY = '5b48d68ce066834a01e3b108a9e5891a'
    url = base_url + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()

    # Konversi suhu
    temp_kelvin = response['main']['temp']
    temp_celcius = kelvin_to_celcius(temp_kelvin)
    feels_like_kelvin = response['main']['feels_like']
    feels_like_celcius = kelvin_to_celcius(feels_like_kelvin)
    description = response['weather'][0]['description']

    # Konversi suhu ke kategori
    temperature = convert_temperature(temp_celcius)

    # Load dataset
    # Dataset harus sudah tersedia dalam variabel 'all'
    ds = all[['product_id', 'gender_product', 'temperature', 'usage', 'productDisplayName', 'home_location', 'link']]

    # Pisahkan fitur dan target
    features = ds[['gender_product', 'temperature']]
    targets = ds['product_id']

    # Konversi fitur kategorikal ke numerikal menggunakan One-Hot Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    features_onehot = onehot_encoder.fit_transform(features)

    # Kombinasi fitur untuk query
    query_vector = onehot_encoder.transform(np.array([gender_product, temperature]).reshape(1, -1))

    # Hitung kesamaan kosinus antara query dan semua data
    similarities = cosine_similarity(query_vector, features_onehot)

    # Dapatkan N tetangga terdekat
    top_n = 5
    sorted_indices = similarities.argsort()[0][-top_n:]

    # Dapatkan rekomendasi product ID
    recommended_outfits = targets.iloc[sorted_indices].tolist()

    return recommended_outfits,
