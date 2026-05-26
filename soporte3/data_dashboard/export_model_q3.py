"""
export_model_q3.py
Run once locally to train the Q3 neural network and save all artifacts
needed by the dashboard.  Mirrors Pregunta_3.ipynb preprocessing exactly.

Usage:
    python export_model_q3.py

Outputs (in ./model_q3/):
    model_q3.keras            - Trained Keras model (5-output regression)
    feature_columns_q3.json   - Ordered list of 455 feature names
    model_info_q3.json        - Per-subject metrics + target stats
    background_mean_q3.npy    - Training-set mean (for group-ablation SHAP)
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "filtered_icfes_data_cesar.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "model_q3")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COLS = [
    "punt_matematicas",
    "punt_lectura_critica",
    "punt_c_naturales",
    "punt_sociales_ciudadanas",
    "punt_ingles",
]
TARGET_LABELS = [
    "Matemáticas",
    "Lectura Crítica",
    "C. Naturales",
    "Soc. y Ciudadanas",
    "Inglés",
]

tf.random.set_seed(42)
np.random.seed(42)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
print(f"  Shape: {df.shape}")

# ── 2. Preprocessing (mirrors Pregunta_3.ipynb exactly) ───────────────────────
print("Preprocessing...")

df = df.astype({
    "periodo": "float32",
    "cole_area_ubicacion": "category",
    "cole_bilingue": "category",
    "cole_calendario": "category",
    "cole_caracter": "category",
    "cole_cod_dane_establecimiento": "category",
    "cole_cod_dane_sede": "category",
    "cole_cod_mcpio_ubicacion": "category",
    "cole_jornada": "category",
    "cole_naturaleza": "category",
    "estu_cod_reside_mcpio": "category",
    "estu_genero": "category",
    "estu_nacionalidad": "category",
    "fami_cuartoshogar": "category",
    "fami_educacionmadre": "category",
    "fami_educacionpadre": "category",
    "fami_estratovivienda": "category",
    "fami_personashogar": "category",
    "fami_tieneautomovil": "category",
    "fami_tienecomputador": "category",
    "fami_tieneinternet": "category",
    "fami_tienelavadora": "category",
})

binary_map = {
    "cole_area_ubicacion":  {"URBANO": True,     "RURAL": False},
    "cole_bilingue":        {"S": True,           "N": False},
    "cole_calendario":      {"A": True,           "B": False},
    "cole_naturaleza":      {"OFICIAL": True,     "NO OFICIAL": False},
    "estu_genero":          {"M": True,           "F": False},
    "fami_tieneautomovil":  {"Si": True,          "No": False},
    "fami_tienecomputador": {"Si": True,          "No": False},
    "fami_tieneinternet":   {"Si": True,          "No": False},
    "fami_tienelavadora":   {"Si": True,          "No": False},
}
for col, mapping in binary_map.items():
    df[col] = df[col].map(mapping)

df = df.rename(columns={
    "cole_area_ubicacion": "cole_area_urbano",
    "cole_calendario":     "cole_calendario_a",
    "cole_naturaleza":     "cole_oficial",
    "estu_genero":         "estu_masculino",
})

# NOTE: "Diez o más" -> "6+" (not "10+" as in Q1)
cuartos_map = {
    "Uno": "1", "Dos": "2", "Tres": "3", "Cuatro": "4", "Cinco": "5",
    "Seis": "6+", "Seis o mas": "6+", "Siete": "6+",
    "Ocho": "6+", "Nueve": "6+", "Diez o más": "6+",
}
df["fami_cuartoshogar"] = df["fami_cuartoshogar"].map(cuartos_map)

# NOTE: "Doce o más" -> "9 o más" (no "12 o más" category in Q3)
personas_map = {
    "Una": "1 a 2", "Dos": "1 a 2",
    "Tres": "3 a 4", "Cuatro": "3 a 4",
    "Cinco": "5 a 6", "Seis": "5 a 6",
    "Siete": "7 a 8", "Ocho": "7 a 8",
    "Nueve": "9 o más", "Diez": "9 o más",
    "Once": "9 o más", "Doce o más": "9 o más",
}
df["fami_personashogar"] = df["fami_personashogar"].map(personas_map)

categorical_cols = [
    "cole_caracter", "cole_cod_dane_establecimiento", "cole_cod_dane_sede",
    "cole_cod_mcpio_ubicacion", "cole_jornada", "estu_cod_reside_mcpio",
    "estu_nacionalidad", "fami_cuartoshogar", "fami_educacionmadre",
    "fami_educacionpadre", "fami_estratovivienda", "fami_personashogar",
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"  Shape after OHE: {df.shape}")

# ── 3. Feature matrix ──────────────────────────────────────────────────────────
DROP_COLS = ["punt_global"] + TARGET_COLS
X = df.drop(columns=DROP_COLS)
y = df[TARGET_COLS]

feature_columns = X.columns.tolist()
print(f"  Features: {len(feature_columns)}")

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train_df.to_numpy(dtype=np.float32)
X_test  = X_test_df.to_numpy(dtype=np.float32)
y_train = y_train_df.to_numpy(dtype=np.float32)
y_test  = y_test_df.to_numpy(dtype=np.float32)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

# ── 4. Build & train (arch_4: Dense-BN stack, 5 outputs) ──────────────────────
print("Building and training model (arch_4)...")

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)

inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = norm_layer(inputs)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(5, activation="linear")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["mae"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
print("Evaluating...")
y_pred = model.predict(X_test, verbose=0)

per_subject = {}
for i, (col, label) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
    mae  = float(mean_absolute_error(y_test[:, i], y_pred[:, i]))
    rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
    r2   = float(r2_score(y_test[:, i], y_pred[:, i]))
    mean_val = float(y[col].mean())
    std_val  = float(y[col].std())
    per_subject[col] = {
        "label": label,
        "mae": mae, "rmse": rmse, "r2": r2,
        "mean": mean_val, "std": std_val,
    }
    print(f"  {label:22s}  MAE={mae:.3f}  R²={r2:.3f}")

mae_mean  = float(np.mean([v["mae"] for v in per_subject.values()]))
rmse_mean = float(np.mean([v["rmse"] for v in per_subject.values()]))
r2_mean   = float(np.mean([v["r2"] for v in per_subject.values()]))
actual_epochs = len(history.history["loss"])
print(f"\n  Mean  MAE={mae_mean:.3f}  R²={r2_mean:.3f}  Epochs={actual_epochs}")

# ── 6. Save artifacts ──────────────────────────────────────────────────────────
print("Saving artifacts...")

model.save(os.path.join(MODEL_DIR, "model_q3.keras"))
print(f"  model_q3.keras saved")

with open(os.path.join(MODEL_DIR, "feature_columns_q3.json"), "w") as f:
    json.dump(feature_columns, f)
print(f"  feature_columns_q3.json saved ({len(feature_columns)} features)")

model_info = {
    "target_cols":   TARGET_COLS,
    "target_labels": TARGET_LABELS,
    "architecture":  "Dense(256,relu)+BN -> Dense(128,relu)+BN -> Dense(64,relu)+BN -> Dense(5)",
    "n_features":    len(feature_columns),
    "n_train":       X_train.shape[0],
    "n_test":        X_test.shape[0],
    "actual_epochs": actual_epochs,
    "batch_size":    32,
    "learning_rate": 1e-3,
    "metrics": {
        "mae_mean":  mae_mean,
        "rmse_mean": rmse_mean,
        "r2_mean":   r2_mean,
        "per_subject": per_subject,
    },
}
with open(os.path.join(MODEL_DIR, "model_info_q3.json"), "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print(f"  model_info_q3.json saved")

background_mean = X_train.mean(axis=0).astype(np.float32)
np.save(os.path.join(MODEL_DIR, "background_mean_q3.npy"), background_mean)
print(f"  background_mean_q3.npy saved ({len(background_mean)} features)")

print(f"\nDone!  Mean MAE: {mae_mean:.2f} pts  |  R²: {r2_mean:.4f}")
