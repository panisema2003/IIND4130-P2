"""
export_model.py
Run ONCE locally to train the best neural network and save all artifacts needed by the dashboard.

Usage:
    python export_model.py

Outputs (in ./model/ directory):
    model.keras           - Trained Keras model (includes Normalization layer)
    feature_columns.json  - Ordered list of 457 feature names used by the model
    categories.json       - Categorical value mappings for dashboard dropdowns
    model_info.json       - Model metrics and metadata
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "filtered_icfes_data_cesar.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)                                  # default UTF-8, mirrors notebook
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

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
    "punt_ingles": "float32",
    "punt_matematicas": "float32",
    "punt_sociales_ciudadanas": "float32",
    "punt_c_naturales": "float32",
    "punt_lectura_critica": "float32",
    "punt_global": "float32",
})
print(f"  Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING  (mirrors question1.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────
print("Preprocessing...")

# 2a. Binary encoding
binary_dict = {
    "cole_area_ubicacion": {"URBANO": True, "RURAL": False},
    "cole_bilingue":       {"S": True,  "N": False},
    "cole_calendario":     {"A": True,  "B": False},
    "cole_naturaleza":     {"OFICIAL": True, "NO OFICIAL": False},
    "estu_genero":         {"M": True,  "F": False},
    "fami_tieneautomovil": {"Si": True, "No": False},
    "fami_tienecomputador":{"Si": True, "No": False},
    "fami_tieneinternet":  {"Si": True, "No": False},
    "fami_tienelavadora":  {"Si": True, "No": False},
}
for col, mapping in binary_dict.items():
    df[col] = df[col].map(mapping).astype("bool")

# 2b. Rename binary columns to meaningful names
df = df.rename(columns={
    "cole_area_ubicacion": "cole_area_urbano",
    "cole_calendario":     "cole_calendario_a",
    "cole_naturaleza":     "cole_oficial",
    "estu_genero":         "estu_masculino",
})

# 2c. Ordinal-like text mappings (then OHE'd)
# Mirrors question1.ipynb exactly
cuartos_map = {
    "Uno": "1", "Dos": "2", "Tres": "3", "Cuatro": "4", "Cinco": "5",
    "Seis": "6+", "Seis o mas": "6+", "Siete": "6+", "Ocho": "6+",
    "Nueve": "6+", "Diez o más": "10+",
}
df["fami_cuartoshogar"] = df["fami_cuartoshogar"].map(cuartos_map)

personas_map = {
    "Una": "1 a 2", "Dos": "1 a 2", "Tres": "3 a 4", "Cuatro": "3 a 4",
    "Cinco": "5 a 6", "Seis": "5 a 6", "Siete": "7 a 8", "Ocho": "7 a 8",
    "Nueve": "9 o más", "Diez": "9 o más", "Once": "9 o más",
    "Doce": "12 o más", "Doce o más": "12 o más",
}
df["fami_personashogar"] = df["fami_personashogar"].map(personas_map)

# 2d. One-hot encode remaining categoricals
ohe_columns = [
    "cole_caracter",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_mcpio_ubicacion",
    "cole_jornada",
    "estu_cod_reside_mcpio",
    "estu_nacionalidad",
    "fami_cuartoshogar",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "fami_personashogar",
]
df = pd.get_dummies(df, columns=ohe_columns, drop_first=True)

# 2e. Drop score columns (except punt_global which is the target)
score_cols = [
    "punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas",
    "punt_c_naturales", "punt_lectura_critica",
]
df = df.drop(columns=score_cols)

print(f"  Features after preprocessing: {df.shape[1] - 1}")  # -1 for punt_global

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=["punt_global"])
y = df["punt_global"]

# Save feature column names (critical for prediction alignment)
feature_columns = X.columns.tolist()
print(f"  Total features: {len(feature_columns)}")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train_df.to_numpy(dtype=np.float32)
X_test  = X_test_df.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32)
y_test  = y_test.to_numpy(dtype=np.float32)

# Background mean for group-ablation explanations (SHAP-like)
background_mean = X_train.mean(axis=0).astype(np.float32)

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD & TRAIN MODEL  (arch_D_deep + full_457 + large_batch)
# ─────────────────────────────────────────────────────────────────────────────
print("Building and training model...")

# Normalization layer (adapts to training data)
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)

inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = norm_layer(inputs)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["mae"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=512,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print("Evaluating...")
y_pred = model.predict(X_test, batch_size=512, verbose=0).flatten()
mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))
actual_epochs = len(history.history["loss"])
print(f"  MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | Epochs={actual_epochs}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("Saving artifacts...")

# Model
model_path = os.path.join(MODEL_DIR, "model.keras")
model.save(model_path)
print(f"  Model saved to {model_path}")

# Background mean
np.save(os.path.join(MODEL_DIR, "background_mean.npy"), background_mean)
print(f"  background_mean.npy saved ({len(background_mean)} features)")

# Feature columns
with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_columns, f)
print(f"  feature_columns.json saved ({len(feature_columns)} features)")

# Model info
model_info = {
    "experiment": "icfes_punt_global_nn",
    "run_name":   "p3_arch_D_deep_full_457_large_batch",
    "architecture": "Dense(256,relu)+BN+Drop(0.3) -> Dense(128,relu)+BN+Drop(0.2) -> Dense(64,relu)+Drop(0.1) -> Dense(32,relu) -> Dense(1)",
    "n_features":   len(feature_columns),
    "n_train":      X_train.shape[0],
    "n_test":       X_test.shape[0],
    "actual_epochs": actual_epochs,
    "batch_size":   512,
    "learning_rate": 1e-3,
    "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    "target": "punt_global",
    "target_range": [float(y.min()), float(y.max())],
    "target_mean":  float(y.mean()),
    "target_std":   float(y.std()),
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)
print(f"  model_info.json saved")

print("\nDone! All artifacts saved to ./model/")
print(f"  MAE: {mae:.2f} points  |  R²: {r2:.4f}")
