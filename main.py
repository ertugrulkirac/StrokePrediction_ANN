# basit_mlp_stroke.py
# pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Veri kümesini oku
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 2. Gereksiz sütunları kaldır
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 3. Eksik verileri temizle
df = df.dropna()

# 4. Kategorik değişkenleri dönüştür
df = pd.get_dummies(df, drop_first=True)

# 5. Hedef ve girdileri ayır
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 6. Eğitim/Test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Normalizasyon
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Basit MLP modeli
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu',
                    solver='adam', max_iter=200, random_state=42)

# 9. Modeli eğit
mlp.fit(X_train, y_train)

# 10. Tahmin ve sonuçlar
y_pred = mlp.predict(X_test)

print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
