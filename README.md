Sosyal Yardım Önceliklendirme Sistemi (AI-Driven Social Aid Decision Support)
Bu proje, sosyo-ekonomik verileri kullanarak sosyal yardım ihtiyacı olan bireyleri tespit eden ve adil bir önceliklendirme yapan makine öğrenmesi tabanlı bir karar destek sistemidir.

🚀 Proje Hakkında
Sınırlı kamu kaynaklarının en doğru ve ihtiyaç sahibi kişilere ulaştırılmasını hedefleyen bu çalışma, GSS (General Social Survey) veri seti üzerinde eğitilmiştir. Model, bireyleri ihtiyaç durumlarına göre 4 farklı kademede (Kritik, Yüksek, İnceleme, Standart) sınıflandırır.

🛠️ Teknik Özellikler
Model: Random Forest Classifier

IQR yöntemiyle aykırı değerler (outliers) baskılandı.

Eksik veriler (NaN) istatistiksel yöntemlerle temizlendi.

Performans: Model başarısı yalnızca Accuracy değil, sosyal hedefler doğrultusunda PR-AUC (Precision-Recall Curve) ve Recall odaklı değerlendirilmiştir.

📊 Önemli Bulgular
Gelir Trendi: Yaş ile gelir arasında pozitif bir trend gözlemlenmiş, genç yaş gruplarının sosyal destek açısından daha kırılgan olduğu saptanmıştır.

Bölgesel Dağılım: Kuzeydoğu ve Batı bölgelerinde ortalama gelirin daha yüksek olduğu, Güney bölgelerinde ise yardım ihtiyacının yoğunlaştığı tespit edilmiştir.

💻 Kurulum ve Çalıştırma
Projeyi yerelinizde çalıştırmak için:

Depoyu klonlayın: git clone <repo-linki>

Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt

Uygulamayı başlatın: streamlit run app.py

⚖️ Etik Beyan
Bu sistem bir karar verici değil, sosyal hizmet uzmanlarına yardımcı bir karar destek aracıdır. Nihai kararların her zaman insan denetiminde (Human-in-the-loop) verilmesi önerilir.
