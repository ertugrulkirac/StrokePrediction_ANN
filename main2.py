# basit_mlp_keras.py
# pip install tensorflow pandas scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Veri kümesini yükle
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 2. Gereksiz sütunları kaldır
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 3. Eksik verileri temizle
df = df.dropna()

# 4. Kategorik değişkenleri one-hot encode et
df = pd.get_dummies(df, drop_first=True)

# 5. Girdiler ve hedef ayrımı
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 6. Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Model oluşturma
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # binary classification için sigmoid çıkış
])

# 9. Model derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 10. Model eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 11. Performans değerlendirmesi
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test doğruluğu: {acc:.4f}")
print(f"Test kaybı: {loss:.4f}")

# 12. Eğitim grafikleri
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# 13. Tahmin örneği
y_pred = (model.predict(X_test) > 0.5).astype("int32")
