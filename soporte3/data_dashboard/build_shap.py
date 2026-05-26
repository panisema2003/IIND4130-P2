"""
build_shap.py
Compute global SHAP feature-group importances for the trained neural network.
Run once after export_model.py to generate model/shap_global.json.

Uses shap.GradientExplainer on a background sample of 100 training rows
and explains 300 held-out test rows.  Aggregates mean |SHAP| by feature group.

Usage:
    python build_shap.py

Output:
    model/shap_global.json  - list of {label, importance} sorted descending
"""

import json
import os

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_utils import FEATURE_GROUPS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "filtered_icfes_data_cesar.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "model")

# ── 1. Load model + feature columns ───────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    feature_columns = json.load(f)
print(f"  {len(feature_columns)} features")

# ── 2. Preprocessing (mirrors export_model.py exactly) ────────────────────────
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
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

binary_dict = {
    "cole_area_ubicacion": {"URBANO": True,      "RURAL": False},
    "cole_bilingue":       {"S": True,            "N": False},
    "cole_calendario":     {"A": True,            "B": False},
    "cole_naturaleza":     {"OFICIAL": True,      "NO OFICIAL": False},
    "estu_genero":         {"M": True,            "F": False},
    "fami_tieneautomovil": {"Si": True,           "No": False},
    "fami_tienecomputador":{"Si": True,           "No": False},
    "fami_tieneinternet":  {"Si": True,           "No": False},
    "fami_tienelavadora":  {"Si": True,           "No": False},
}
for col, mapping in binary_dict.items():
    df[col] = df[col].map(mapping).astype("bool")

df = df.rename(columns={
    "cole_area_ubicacion": "cole_area_urbano",
    "cole_calendario":     "cole_calendario_a",
    "cole_naturaleza":     "cole_oficial",
    "estu_genero":         "estu_masculino",
})

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

ohe_columns = [
    "cole_caracter", "cole_cod_dane_establecimiento", "cole_cod_dane_sede",
    "cole_cod_mcpio_ubicacion", "cole_jornada", "estu_cod_reside_mcpio",
    "estu_nacionalidad", "fami_cuartoshogar", "fami_educacionmadre",
    "fami_educacionpadre", "fami_estratovivienda", "fami_personashogar",
]
df = pd.get_dummies(df, columns=ohe_columns, drop_first=True)
df = df.drop(columns=[
    "punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas",
    "punt_c_naturales", "punt_lectura_critica",
])

X = df.drop(columns=["punt_global"]).reindex(columns=feature_columns, fill_value=0)
y = df["punt_global"]

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np  = X_test.to_numpy(dtype=np.float32)
print(f"  Train {X_train_np.shape} | Test {X_test_np.shape}")

# ── 3. SHAP GradientExplainer ─────────────────────────────────────────────────
np.random.seed(42)
bg_idx  = np.random.choice(len(X_train_np), size=100, replace=False)
exp_idx = np.random.choice(len(X_test_np),  size=300, replace=False)

background = X_train_np[bg_idx]
X_explain  = X_test_np[exp_idx]

print("Computing SHAP values (GradientExplainer, 300 samples)...")
explainer   = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

# Handle both (n, f) and [(n, f)] return shapes
if isinstance(shap_values, list):
    shap_arr = np.array(shap_values[0])
else:
    shap_arr = np.array(shap_values)

# Squeeze trailing dim if shape is (n, f, 1)
if shap_arr.ndim == 3:
    shap_arr = shap_arr[:, :, 0]

print(f"  SHAP array shape: {shap_arr.shape}")
mean_abs_shap = np.abs(shap_arr).mean(axis=0)   # (n_features,)

# ── 4. Aggregate by feature group ─────────────────────────────────────────────
feat_idx = {c: i for i, c in enumerate(feature_columns)}

group_importances = {}
for label, group_def in FEATURE_GROUPS:
    indices = []
    if isinstance(group_def, list):
        for feat_name in group_def:
            if feat_name in feat_idx:
                indices.append(feat_idx[feat_name])
    else:
        prefix = group_def
        for feat_name in feature_columns:
            if feat_name.startswith(prefix):
                indices.append(feat_idx[feat_name])
    if indices:
        group_importances[label] = float(mean_abs_shap[indices].sum())

sorted_groups = sorted(group_importances.items(), key=lambda x: x[1], reverse=True)

output = [{"label": lbl, "importance": imp} for lbl, imp in sorted_groups]

out_path = os.path.join(MODEL_DIR, "shap_global.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(output)} groups to {out_path}")
print("\nTop 10 global feature group importances:")
for g in output[:10]:
    bar = "#" * int(g["importance"] * 3)
    print(f"  {g['importance']:6.2f} pts  {g['label']}")
