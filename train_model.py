"""Veriyi incele, modelleri eğit ve rapor için sonuçları üret."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


KOK = Path(__file__).resolve().parent
VERI_DOSYASI = KOK / "online_shoppers_intention.csv"
CIKTI_KLASORU = KOK / "outputs"
CIKTI_KLASORU.mkdir(exist_ok=True)

ham_veri = pd.read_csv(VERI_DOSYASI)
# Oturum kimliği bulunmadığı için birebir aynı satırların ayrı oturumlar
# olup olmadığını bilemeyiz. Kopyaların eğitim ve testte birlikte yer almasını
# önlemek amacıyla bunları veri bölünmeden önce çıkarıyoruz.
veri = ham_veri.drop_duplicates().reset_index(drop=True)
hedef = veri["Revenue"].astype(int)
girdiler = veri.drop(columns="Revenue")

# Sayısal sütunları ölçekleme yöntemi farklı olduğu için ayrıca belirtiyoruz.
sayisal_sutunlar = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
kategorik_sutunlar = [sutun for sutun in girdiler.columns if sutun not in sayisal_sutunlar]

# İlk bölme %80/%20, ikinci bölme kalan %80'in %75/%25'idir.
# Böylece eğitim/doğrulama/test oranları %60/%20/%20 olur.
girdi_gecici, girdi_test, hedef_gecici, hedef_test = train_test_split(
    girdiler, hedef, test_size=0.20, random_state=42, stratify=hedef
)
girdi_egitim, girdi_dogrulama, hedef_egitim, hedef_dogrulama = train_test_split(
    girdi_gecici, hedef_gecici, test_size=0.25, random_state=42, stratify=hedef_gecici
)

# Dönüşümler Pipeline içinde eğitildiği için doğrulama ve test verisinden
# eğitim aşamasına bilgi sızmaz.
lojistik_on_isleme = ColumnTransformer(
    [
        ("sayisal", StandardScaler(), sayisal_sutunlar),
        ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik_sutunlar),
    ]
)
orman_on_isleme = ColumnTransformer(
    [
        ("sayisal", "passthrough", sayisal_sutunlar),
        ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik_sutunlar),
    ]
)
modeller = {
    "Çoğunluk modeli": DummyClassifier(strategy="most_frequent"),
    "Lojistik regresyon": Pipeline(
        [
            ("on_isleme", lojistik_on_isleme),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("on_isleme", orman_on_isleme),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    ),
}


def modeli_degerlendir(model, girdi_verisi, gercek_siniflar):
    """Satın alma sınıfını (1) esas alarak başarı ölçülerini hesapla."""
    tahminler = model.predict(girdi_verisi)
    hata_matrisi = confusion_matrix(gercek_siniflar, tahminler, labels=[0, 1])
    olculer = {
        "accuracy": float(accuracy_score(gercek_siniflar, tahminler)),
        "precision": float(precision_score(gercek_siniflar, tahminler, zero_division=0)),
        "recall": float(recall_score(gercek_siniflar, tahminler, zero_division=0)),
        "f1": float(f1_score(gercek_siniflar, tahminler, zero_division=0)),
        "confusion_matrix": hata_matrisi.tolist(),
    }
    if hasattr(model, "predict_proba"):
        satin_alma_olasiligi = model.predict_proba(girdi_verisi)[:, 1]
        olculer["roc_auc"] = float(roc_auc_score(gercek_siniflar, satin_alma_olasiligi))
    return olculer


sonuclar = {}
for model_adi, model in modeller.items():
    model.fit(girdi_egitim, hedef_egitim)
    sonuclar[model_adi] = {
        "validation": modeli_degerlendir(model, girdi_dogrulama, hedef_dogrulama),
        "test": modeli_degerlendir(model, girdi_test, hedef_test),
    }

# Model seçimi yalnızca doğrulama F1 değerine dayanır; test kümesi bu
# kararda kullanılmaz.
secilen_model_adi = max(
    (model_adi for model_adi in modeller if model_adi != "Çoğunluk modeli"),
    key=lambda model_adi: sonuclar[model_adi]["validation"]["f1"],
)
secilen_model = modeller[secilen_model_adi]
test_tahminleri = secilen_model.predict(girdi_test)
hata_profilleri = {}
# Hata türleri için test oturumlarının ortanca özelliklerine bakıyoruz.
for grup_adi, maske in {
    "true_positive": (hedef_test.to_numpy() == 1) & (test_tahminleri == 1),
    "false_negative": (hedef_test.to_numpy() == 1) & (test_tahminleri == 0),
    "false_positive": (hedef_test.to_numpy() == 0) & (test_tahminleri == 1),
    "true_negative": (hedef_test.to_numpy() == 0) & (test_tahminleri == 0),
}.items():
    oturumlar = girdi_test.loc[maske]
    hata_profilleri[grup_adi] = {
        "count": len(oturumlar),
        "median_page_values": float(oturumlar["PageValues"].median()),
        "median_product_pages": float(oturumlar["ProductRelated"].median()),
        "median_exit_rates": float(oturumlar["ExitRates"].median()),
    }

# PageValues'ın gerçek tahmin anında hazır olup olmayacağı belirsizdir.
# Modelin bu değişkene bağımlılığını görmek için aynı deneyi onsuz tekrarlıyoruz.
pagevalues_olmadan = Pipeline(
    [
        (
            "on_isleme",
            ColumnTransformer(
                [
                    ("sayisal", "passthrough", [sutun for sutun in sayisal_sutunlar if sutun != "PageValues"]),
                    ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik_sutunlar),
                ]
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)
pagevalues_olmadan.fit(girdi_egitim, hedef_egitim)

ozet = {
    "source_rows": len(ham_veri),
    "duplicate_rows_removed": len(ham_veri) - len(veri),
    "analysis_rows": len(veri),
    "feature_count": len(sayisal_sutunlar) + len(kategorik_sutunlar),
    "missing_cells": int(ham_veri.isna().sum().sum()),
    "positive_count": int(hedef.sum()),
    "positive_rate": float(hedef.mean()),
    "split": {
        "train": len(hedef_egitim),
        "validation": len(hedef_dogrulama),
        "test": len(hedef_test),
        "test_positive": int(hedef_test.sum()),
    },
    "visitor_type": veri.groupby("VisitorType")["Revenue"].agg(["count", "mean"]).to_dict("index"),
    "month": veri.groupby("Month")["Revenue"].agg(["count", "mean"]).to_dict("index"),
    "product_related_median": veri.groupby("Revenue")["ProductRelated"].median().to_dict(),
    "page_values_median": veri.groupby("Revenue")["PageValues"].median().to_dict(),
    "selected_by_validation_f1": secilen_model_adi,
    "models": sonuclar,
    "error_profiles": hata_profilleri,
    "random_forest_without_page_values": {
        "validation": modeli_degerlendir(pagevalues_olmadan, girdi_dogrulama, hedef_dogrulama),
        "test": modeli_degerlendir(pagevalues_olmadan, girdi_test, hedef_test),
    },
}
(CIKTI_KLASORU / "results.json").write_text(
    json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8"
)

# Aşağıdaki dört grafik rapordaki şekillerle aynı dosyalardır.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
colors = ["#456B89", "#E99B54"]

# Şekil 1: Sınıflar arasındaki dengesizliği göster.
fig, ax = plt.subplots(figsize=(6.6, 3.4))
counts = hedef.value_counts().sort_index()
bars = ax.bar(["Satın alma yok", "Satın alma var"], counts.values, color=colors)
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, count + 70, f"{count:,}", ha="center")
ax.set(ylabel="Oturum sayısı", title="Tekrar eden kayıtlar çıkarıldıktan sonra sınıf dağılımı")
ax.set_ylim(0, counts.max() * 1.14)
fig.tight_layout()
fig.savefig(CIKTI_KLASORU / "class_balance.png", dpi=180)
plt.close(fig)

# Şekil 2: Ziyaretçi türleri arasındaki gözlenen oran farkını göster.
group = veri.groupby("VisitorType")["Revenue"].agg(["count", "mean"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(6.6, 3.4))
visitor_labels = {"Returning_Visitor": "Geri dönen", "Other": "Diğer", "New_Visitor": "Yeni"}
bars = ax.bar([visitor_labels[label] for label in group.index], group["mean"] * 100, color="#456B89")
for bar, count in zip(bars, group["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"n={count}", ha="center", fontsize=9)
ax.set(ylabel="Satın alma oranı (%)", title="Ziyaretçi türüne göre satın alma oranı")
ax.set_ylim(0, max(group["mean"] * 100) * 1.23)
fig.tight_layout()
fig.savefig(CIKTI_KLASORU / "visitor_type.png", dpi=180)
plt.close(fig)

# Şekil 3: Ürün sayfası ziyaretlerinin iki sınıftaki dağılımını göster.
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for label, color in [(False, colors[0]), (True, colors[1])]:
    vals = np.log1p(veri.loc[veri.Revenue == label, "ProductRelated"])
    ax.hist(vals, bins=35, density=True, alpha=0.58, color=color, label="Satın alma var" if label else "Satın alma yok")
ax.set(xlabel="log(1 + ziyaret edilen ürün sayfası)", ylabel="Yoğunluk", title="Sonuca göre ürün sayfası ziyaretleri")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(CIKTI_KLASORU / "product_pages.png", dpi=180)
plt.close(fig)

# Şekil 4: Seçilen modelin test hatalarını dört grupta göster.
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ConfusionMatrixDisplay(
    confusion_matrix=np.asarray(sonuclar[secilen_model_adi]["test"]["confusion_matrix"]),
    display_labels=["Satın alma yok", "Satın alma var"],
).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Test karmaşıklık matrisi: Random Forest")
ax.set_xlabel("Tahmin")
ax.set_ylabel("Gerçek")
fig.tight_layout()
fig.savefig(CIKTI_KLASORU / "confusion_matrix.png", dpi=180)
plt.close(fig)

print(f"Analiz tamamlandı. Seçilen model: {secilen_model_adi}")
print(f"Test F1: {sonuclar[secilen_model_adi]['test']['f1']:.3f}")
print(f"Ayrıntılı ölçümler: {CIKTI_KLASORU / 'results.json'}")
