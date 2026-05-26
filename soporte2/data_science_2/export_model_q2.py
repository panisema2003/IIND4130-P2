"""
export_model_q2.py
Entrena el clasificador bilingue (Q2) y exporta todos los artefactos a
data_dashboard/model_q2/
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as sk_cw
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../data/clean_icfes_data_cesar.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "../data_dashboard/model_q2")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Carga y limpieza
print("Cargando datos...")
df = pd.read_csv(DATA_PATH)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df = df.dropna(subset=["cole_bilingue"])
df.columns = df.columns.str.lower()

features_num = [
    "punt_lectura_critica", "punt_matematicas", "punt_sociales_ciudadanas",
    "punt_c_naturales", "punt_ingles", "punt_global", "fami_estratovivienda",
]
features_cat = [
    "cole_naturaleza", "cole_jornada", "estu_genero",
    "fami_tieneinternet", "fami_tienecomputador",
]

df_clean = df[features_num + features_cat + ["cole_bilingue"]].copy()
df_clean = df_clean.dropna()

# Extraer estrato numerico de cadenas tipo "Estrato N"
df_clean["fami_estratovivienda"] = (
    df_clean["fami_estratovivienda"]
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)
df_clean = df_clean.dropna(subset=["fami_estratovivienda"])

# Normalizar encoding corrupto MANANA para nombres de columna limpios
df_clean["cole_jornada"] = df_clean["cole_jornada"].apply(
    lambda x: "MANANA" if isinstance(x, str) and x.upper().startswith("MA") and x.upper().endswith("ANA") and "U" not in x.upper() else x
)

print(f"  Registros: {len(df_clean):,}  |  Bilingues: {(df_clean['cole_bilingue']=='S').sum():,}")

# 2. Codificacion
df_encoded = pd.get_dummies(df_clean, columns=features_cat, drop_first=True, dtype=float)

feature_cols = [
    c for c in df_encoded.columns
    if c in features_num or any(cat in c for cat in features_cat)
]
X = df_encoded[feature_cols].astype(float)
y = (df_encoded["cole_bilingue"] == "S").astype(int)

print(f"  Variables ({len(feature_cols)}): {feature_cols}")

# 3. Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr = X_train.values.astype(np.float32)
X_te = X_test.values.astype(np.float32)

# 4. Pesos de clase para compensar desbalance (~2% bilingues)
w = sk_cw.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {0: float(w[0]), 1: float(w[1])}
print(f"  Pesos de clase: {class_weights}")

# 5. Normalizacion
norm = tf.keras.layers.Normalization()
norm.adapt(X_tr)

# 6. Arquitectura: medium [128,64,32] seleccionada por busqueda en MLflow
n = X_tr.shape[1]
model = Sequential([
    Input(shape=(n,)),
    norm,
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)
model.summary()

# 7. Entrenamiento
print("Entrenando modelo...")
early = EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)
model.fit(
    X_tr, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early],
    verbose=1,
)

# 8. Evaluacion
y_prob = model.predict(X_te, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

metrics = {
    "accuracy":  float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
    "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    "auc_roc":   float(roc_auc_score(y_test, y_prob)),
}
print("Metricas:", {k: f"{v:.4f}" for k, v in metrics.items()})

# Umbral optimo maximizando F1
thresholds = np.arange(0.05, 0.95, 0.01)
f1s = [f1_score(y_test, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
best_thr = float(thresholds[np.argmax(f1s)])
print(f"  Umbral optimo (F1-max): {best_thr:.2f}")

# 9. Guardar artefactos
model.save(os.path.join(OUT_DIR, "model.keras"))

with open(os.path.join(OUT_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "model_info.json"), "w", encoding="utf-8") as f:
    json.dump({
        "task":         "binary_classification",
        "target":       "cole_bilingue",
        "target_pos":   "S",
        "n_features":   len(feature_cols),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "architecture": "medium [128,64,32] dropout=[0.4,0.3,0.2] lr=0.0001",
        "metrics":      metrics,
        "class_dist":   {
            "no_bilingue": int((y == 0).sum()),
            "bilingue":    int((y == 1).sum()),
        },
    }, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT_DIR, "class_weights.json"), "w") as f:
    json.dump({"0": class_weights[0], "1": class_weights[1]}, f)

with open(os.path.join(OUT_DIR, "threshold.json"), "w") as f:
    json.dump({"threshold": best_thr, "default": 0.5}, f)

np.save(os.path.join(OUT_DIR, "background_mean.npy"), X_tr.mean(axis=0))

print("Artefactos guardados en", OUT_DIR)
for fname in sorted(os.listdir(OUT_DIR)):
    print(f"  {fname}")
