"""
build_colegios.py
Run once locally (before Docker build) to generate model/colegios.json —
a lookup table mapping DANE establishment codes to school names and attributes.

Usage:
    python build_colegios.py

Output:
    model/colegios.json
"""

import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV  = os.path.join(SCRIPT_DIR, "..", "data", "clean_icfes_data_cesar.csv")
FILT_CSV   = os.path.join(SCRIPT_DIR, "..", "data", "filtered_icfes_data_cesar.csv")
OUTPUT     = os.path.join(SCRIPT_DIR, "model", "colegios.json")

os.makedirs(os.path.join(SCRIPT_DIR, "model"), exist_ok=True)

COLE_COLS = [
    "cole_cod_dane_establecimiento",
    "cole_nombre_establecimiento",
    "cole_cod_mcpio_ubicacion",
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
    "cole_caracter",
    "cole_jornada",
    "cole_naturaleza",
]

print("Loading CSVs...")
df_filt  = pd.read_csv(FILT_CSV,  encoding="utf-8")
df_clean = pd.read_csv(CLEAN_CSV, encoding="utf-8")

# Strip stray whitespace/quotes from all string columns
for df in (df_filt, df_clean):
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip().str.strip("'"))

# Schools present in the training data (filtered CSV)
training_codes = set(df_filt["cole_cod_dane_establecimiento"].dropna().astype(str))
print(f"  Unique schools in training data: {len(training_codes)}")

# Use clean CSV for names and attributes; restrict to training schools
df_clean["_code_str"] = df_clean["cole_cod_dane_establecimiento"].astype(str)
# Only include schools present in filtered (training) data — others have no model coefficient
df_clean = df_clean[df_clean["_code_str"].isin(training_codes)]
df_schools = df_clean[COLE_COLS].copy()
df_schools["cole_cod_dane_establecimiento"] = df_schools[
    "cole_cod_dane_establecimiento"
].astype(str)


def _mode(series):
    m = series.dropna().mode()
    return m.iloc[0] if not m.empty else None


colegios = {}
for dane_code, grp in df_schools.groupby("cole_cod_dane_establecimiento"):
    nombre     = _mode(grp["cole_nombre_establecimiento"])
    if nombre is None:
        continue
    municipio  = int(_mode(grp["cole_cod_mcpio_ubicacion"]))
    area       = _mode(grp["cole_area_ubicacion"]) or "URBANO"
    bilingue   = _mode(grp["cole_bilingue"]) or "N"
    calendario = _mode(grp["cole_calendario"]) or "A"
    caracter   = _mode(grp["cole_caracter"])  or "ACADÉMICO"
    jornada    = _mode(grp["cole_jornada"])   or "UNICA"
    naturaleza = _mode(grp["cole_naturaleza"]) or "OFICIAL"

    colegios[dane_code] = {
        "nombre":    nombre,
        "municipio": municipio,
        "area_urbano":  area == "URBANO",
        "bilingue":     bilingue == "S",
        "calendario_a": calendario == "A",
        "caracter":     caracter,   # raw string for OHE match
        "jornada":      jornada,    # raw string for OHE match
        "oficial":      naturaleza == "OFICIAL",
    }

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(colegios, f, ensure_ascii=False, indent=2)

print(f"Saved {len(colegios)} schools to {OUTPUT}")
print(f"  ({len(training_codes)} in training data, {len(colegios) - len(training_codes)} additional)")
