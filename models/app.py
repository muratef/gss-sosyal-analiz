import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Dosyaları yükle
model = joblib.load('best_rf_model.pkl')
imputer_num = joblib.load('imputer_num.pkl')
imputer_cat = joblib.load('imputer_cat.pkl')

# 2. Modelin beklediği tam liste ve sıra (Kopyaladığın liste)
all_features = ['year', 'income', 'educ', 'health', 'happy', 'prestg10', 
                'paeduc', 'age', 'sex', 'race', 'marital', 'polviews', 
                'trust', 'region']

st.set_page_config(page_title="Sosyal Yardım Analiz Sistemi", layout="centered")
st.title("💰 İhtiyaç Sahibi Tespit Paneli")

# 3. Sidebar Giriş Alanları
st.sidebar.header("Kişi Bilgileri")
age = st.sidebar.number_input("Yaş", 18, 100, 30)
educ = st.sidebar.number_input("Eğitim Yılı", 0, 20, 12)
prestg10 = st.sidebar.slider("Mesleki Prestij", 10, 80, 40)
health = st.sidebar.selectbox("Sağlık (1:Mükemmel - 4:Kötü)", [1, 2, 3, 4])
sex = st.sidebar.selectbox("Cinsiyet (1:E, 2:K)", [1, 2])
marital = st.sidebar.selectbox("Medeni Durum (1-5)", [1, 2, 3, 4, 5])
region = st.sidebar.selectbox("Bölge (1-9)", list(range(1, 10)))

# 4. Tahmin Butonu
if st.button("Kişiyi Analiz Et"):
    # Arayüzde sormadığımız değişkenler için "ortalama/güvenli" varsayılanlar
    input_dict = {
        'age': age, 'educ': educ, 'prestg10': prestg10, 'health': health,
        'sex': sex, 'marital': marital, 'region': region,
        'year': 2024,      # Güncel yıl
        'income': 12,      # Orta seviye gelir kodu
        'happy': 2,       # Orta mutluluk seviyesi
        'paeduc': 10,      # Baba eğitim yılı ortalaması
        'race': 1,        # Genel kategori
        'polviews': 4,    # Orta siyasi görüş
        'trust': 2        # Genel güven seviyesi
    }
    
    # DataFrame oluştur ve sütun sırasını modelin beklediği hale getir
    user_data = pd.DataFrame([input_dict])[all_features]
    
    # 5. Imputer İşlemleri (Sütun isimlerine göre ayırarak)
    sayisal = ['year', 'income', 'educ', 'prestg10', 'paeduc', 'age']
    kategorik = ['health', 'happy', 'sex', 'race', 'marital', 'polviews', 'trust', 'region']
    
    user_data[sayisal] = imputer_num.transform(user_data[sayisal])
    user_data[kategorik] = imputer_cat.transform(user_data[kategorik])
    
# ... (tahmin satırına kadar olan kodlar aynı) ...
    prob = model.predict_proba(user_data)[:, 1][0]
    
    st.write("---")
    st.subheader("📊 Analiz Sonucu")

    # Rakamı gizleyip 'Durum' seviyesine çeviriyoruz
    if prob >= 0.5:
        st.error("🔴 DURUM: KRİTİK ÖNCELİKLİ")
        st.info("Sosyal yardım kriterlerinin tamamını karşılıyor. Acil inceleme önerilir.")
    elif prob >= 0.3:
        st.warning("🟡 DURUM: YÜKSEK ÖNCELİKLİ")
        st.info("Sosyal yardım için güçlü göstergeler mevcut. Programa dahil edilmeye aday.")
    elif prob >= 0.15:
        st.info("🔵 DURUM: İNCELEME GEREKEBİLİR")
        st.info("Sınırda bir profil. Ek belgelerle desteklenmesi önerilir.")
    else:
        st.success("🟢 DURUM: STANDART / DÜŞÜK ÖNCELİK")
        st.info("Mevcut kriterlere göre öncelikli grupta yer almamaktadır.")