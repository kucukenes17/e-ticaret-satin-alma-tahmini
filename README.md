# E-ticaret oturumlarında satın alma tahmini

Bu depo, proje yönergesine göre hazırlanmış sınıflandırma çalışmasını, Türkçe açıklamalı kodu ve dört sayfalık raporu içerir.

**Rapor:** [PDF](E_ticaret_satin_alma_tahmini_raporu.pdf) · [düzenlenebilir Word dosyası](E_ticaret_satin_alma_tahmini_raporu.docx)

## Veri

- Kaynak: C. Sakar ve Y. Kastro, [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), UCI Machine Learning Repository, 2018.
- DOI: https://doi.org/10.24432/C5F88Q
- Lisans: CC BY 4.0.
- Ham veri: `online_shoppers_intention.csv` (12.330 oturum, 17 girdi ve `Revenue` hedefi).

## Yeniden üretim

Python 3.12 ile `requirements.txt` içindeki paketleri kurun. Sonra bu klasörde:

```powershell
python -m pip install -r requirements.txt
python train_model.py
python build_report.py
```

İlk komut `outputs/results.json` dosyasını ve dört grafiği yeniden üretir. İkinci komut bu ölçümlerden `E_ticaret_satin_alma_tahmini_raporu.docx` dosyasını oluşturur.

Yöntem: birebir aynı satırlar bölmeden önce çıkarılır; eğitim/doğrulama/test ayrımı sınıf oranı korunarak %60/%20/%20 yapılır (`random_state=42`). Lojistik regresyon, Random Forest ve çoğunluk sınıfı temel modeli karşılaştırılır. Model seçimi doğrulama F1 değeriyle yapılır; test kümesi son değerlendirmede kullanılır. PageValues olmadan ek bir duyarlılık deneyi de çalıştırılır.

## Kod rehberi

- `train_model.py`: veriyi okur, birebir tekrarları çıkarır, veriyi böler, modelleri eğitir, test sonuçlarını hesaplar ve grafikleri kaydeder.
- `build_report.py`: `outputs/results.json` içindeki ölçümleri ve grafikleri kullanarak Word raporunu üretir.
- `outputs/results.json`: rapordaki sayısal sonuçların kaynağıdır. Değişken adlarının bir kısmı rapor üreticisiyle uyum için İngilizce tutulmuştur; kaynak veri sütunları da UCI'deki özgün adlarıyla kullanılır.

Test kümesinde seçilen Random Forest için accuracy %88,1, precision %59,3, recall %76,7 ve F1 %66,9'dur. `PageValues` çıkarıldığında F1 %40,0'a düşer. Bu nedenle rapor, gerçek zamanlı kullanımı doğrulanmış bir ürün iddiasında bulunmaz.

Kişi veya kurum bilgisi istenirse rapora ayrıca eklenmelidir.
