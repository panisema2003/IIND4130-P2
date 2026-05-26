"""
build_background.py
Run once to generate model/background_mean.npy — the training-set feature mean
used for group-ablation explanations in the dashboard.

Uses the same preprocessing as export_model.py but skips training.

Usage:
    python build_background.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "filtered_icfes_data_cesar.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "model")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_columns.json")
OUT_PATH   = os.path.join(MODEL_DIR, "background_mean.npy")

print("Loading data...")
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

score_cols = [
    "punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas",
    "punt_c_naturales", "punt_lectura_critica",
]
df = df.drop(columns=score_cols)

X = df.drop(columns=["punt_global"])
y = df["punt_global"]

# Load feature_columns saved by export_model.py to ensure alignment
with open(FEAT_PATH) as f:
    feature_columns = json.load(f)

X = X.reindex(columns=feature_columns, fill_value=0)

X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_np = X_train.to_numpy(dtype=np.float32)

background_mean = X_train_np.mean(axis=0).astype(np.float32)
np.save(OUT_PATH, background_mean)
print(f"Saved background_mean.npy ({len(background_mean)} features) -> {OUT_PATH}")
