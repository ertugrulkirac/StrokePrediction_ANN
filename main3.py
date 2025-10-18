# mlp_keras_weights_save.py
# pip install tensorflow pandas scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# === 1. Veri yükleme ===
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
if "id" in df.columns:
    df = df.drop(columns=["id"])
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 2. Model oluşturma ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), name='hidden_1'),
    Dropout(0.3),
    Dense(32, activation='relu', name='hidden_2'),
    Dense(1, activation='sigmoid', name='output')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 3. Eğitim ===
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=30, batch_size=32, verbose=1)

# === 4. Ağırlık ve bias kaydı ===
weights_data = []

for layer in model.layers:
    if len(layer.get_weights()) == 2:  # Dense katmanlarında (W, b)
        W, b = layer.get_weights()
        layer_name = layer.name
        w_flat = W.flatten()
        b_flat = b.flatten()

        # Ağırlıklar için DataFrame
        w_df = pd.DataFrame(w_flat, columns=[f"{layer_name}_weights"])
        b_df = pd.DataFrame(b_flat, columns=[f"{layer_name}_biases"])

        # Katman isimlerine göre CSV oluştur
        w_df.to_csv(f"{layer_name}_weights.csv", index=False)
        b_df.to_csv(f"{layer_name}_biases.csv", index=False)

        weights_data.append({
            "layer": layer_name,
            "weights_shape": W.shape,
            "bias_shape": b.shape
        })

# === 5. Özet rapor ===
summary_df = pd.DataFrame(weights_data)
summary_df.to_csv("model_weights_summary.csv", index=False)

print("Tüm katmanların ağırlık ve bias dosyaları kaydedildi.")
print(summary_df)

# === 6. Modeli kaydet (opsiyonel) ===
model.save("stroke_ann_model.h5")
