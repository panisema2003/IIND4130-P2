"""
model_utils_q3.py
Preprocessing + inference utilities for Q3 (multi-output score predictor).
Mirrors Pregunta_3.ipynb exactly.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Feature groups (reused from Q1 — same input space) ────────────────────────
from model_utils import FEATURE_GROUPS, MUNICIPIOS_CESAR, PERIODOS_LABELS, EDUCATION_LABELS

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

# Colour per subject (consistent across the dashboard)
TARGET_COLORS = {
    "punt_matematicas":        "#2980b9",
    "punt_lectura_critica":    "#27ae60",
    "punt_c_naturales":        "#d35400",
    "punt_sociales_ciudadanas":"#8e44ad",
    "punt_ingles":             "#7B1C2B",
}

# Cuartos categories for Q3 (max "6+", no "10+")
CUARTOS_OPTIONS_Q3 = [
    {"label": "1 cuarto",  "value": "1"},
    {"label": "2 cuartos", "value": "2"},
    {"label": "3 cuartos", "value": "3"},
    {"label": "4 cuartos", "value": "4"},
    {"label": "5 cuartos", "value": "5"},
    {"label": "6 o más",   "value": "6+"},
]

# Personas categories for Q3 (max "9 o más", no "12 o más")
PERSONAS_OPTIONS_Q3 = [
    {"label": "1 a 2",   "value": "1 a 2"},
    {"label": "3 a 4",   "value": "3 a 4"},
    {"label": "5 a 6",   "value": "5 a 6"},
    {"label": "7 a 8",   "value": "7 a 8"},
    {"label": "9 o más", "value": "9 o más"},
]


class ModelLoaderQ3:
    """Loads and caches the Q3 multi-output model + artifacts at startup."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns: list = []
        self.model_info: dict = {}
        self.background_mean: np.ndarray | None = None
        self.available = False
        self._load()

    def _load(self):
        model_path = os.path.join(self.model_dir, "model_q3.keras")
        feat_path  = os.path.join(self.model_dir, "feature_columns_q3.json")
        info_path  = os.path.join(self.model_dir, "model_info_q3.json")
        bg_path    = os.path.join(self.model_dir, "background_mean_q3.npy")

        if not os.path.exists(model_path):
            return  # available stays False

        self.model = tf.keras.models.load_model(model_path)
        with open(feat_path, encoding="utf-8") as f:
            self.feature_columns = json.load(f)
        with open(info_path, encoding="utf-8") as f:
            self.model_info = json.load(f)
        if os.path.exists(bg_path):
            self.background_mean = np.load(bg_path)
        self.available = True

    def predict(self, user_inputs: dict) -> dict:
        """
        Returns dict: {col: float} for each of the 5 target subjects.
        Scores are clipped to [0, 100].
        """
        row = self._build_feature_row(user_inputs)
        x   = row.to_numpy(dtype=np.float32).reshape(1, -1)
        raw = self.model.predict(x, verbose=0)[0]   # shape (5,)
        return {
            col: float(np.clip(raw[i], 0, 100))
            for i, col in enumerate(TARGET_COLS)
        }

    def explain_groups(self, user_inputs: dict, top_n: int = 5) -> list:
        """
        Group ablation: measures mean |impact| across all 5 outputs.
        Returns top_n groups by absolute mean impact, sorted descending.
        Each item: {"label": str, "impact": float}  (mean pts change across subjects)
        """
        if self.background_mean is None:
            return []

        baseline_row  = self._build_feature_row(user_inputs).to_numpy(dtype=np.float32)
        baseline_pred = self.model.predict(baseline_row.reshape(1, -1), verbose=0)[0]  # (5,)
        feat_idx = {c: i for i, c in enumerate(self.feature_columns)}

        ablated_rows  = []
        group_labels  = []

        for label, group_def in FEATURE_GROUPS:
            row = baseline_row.copy()
            if isinstance(group_def, list):
                for feat_name in group_def:
                    if feat_name in feat_idx:
                        i = feat_idx[feat_name]
                        row[i] = self.background_mean[i]
            else:
                for feat_name in self.feature_columns:
                    if feat_name.startswith(group_def):
                        i = feat_idx[feat_name]
                        row[i] = self.background_mean[i]
            ablated_rows.append(row)
            group_labels.append(label)

        batch        = np.array(ablated_rows, dtype=np.float32)
        ablated_preds = self.model.predict(batch, verbose=0)  # (n_groups, 5)

        impacts = []
        for lbl, abl_pred in zip(group_labels, ablated_preds):
            mean_impact = float(np.mean(baseline_pred - abl_pred))
            impacts.append((lbl, mean_impact))

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return [{"label": lbl, "impact": imp} for lbl, imp in impacts[:top_n]]

    def _build_feature_row(self, ui: dict) -> pd.Series:
        """Reconstruct a single preprocessed row aligned to feature_columns_q3."""
        raw: dict = {}

        raw["periodo"] = float(ui["periodo"])

        raw["cole_area_urbano"]     = bool(ui["cole_area_urbano"])
        raw["cole_bilingue"]        = bool(ui["cole_bilingue"])
        raw["cole_calendario_a"]    = bool(ui["cole_calendario_a"])
        raw["cole_oficial"]         = bool(ui["cole_oficial"])
        raw["estu_masculino"]       = bool(ui["estu_masculino"])
        raw["fami_tieneautomovil"]  = bool(ui["fami_tieneautomovil"])
        raw["fami_tienecomputador"] = bool(ui["fami_tienecomputador"])
        raw["fami_tieneinternet"]   = bool(ui["fami_tieneinternet"])
        raw["fami_tienelavadora"]   = bool(ui["fami_tienelavadora"])

        def _set_ohe(prefix: str, value: str):
            raw[f"{prefix}_{value}"] = 1

        _set_ohe("cole_caracter",           str(ui["cole_caracter"]))
        _set_ohe("cole_cod_mcpio_ubicacion", str(int(ui["cole_cod_mcpio_ubicacion"])))
        _set_ohe("estu_cod_reside_mcpio",    str(int(ui["estu_cod_reside_mcpio"])))
        _set_ohe("cole_jornada",             str(ui["cole_jornada"]))
        _set_ohe("estu_nacionalidad",        str(ui["estu_nacionalidad"]))
        _set_ohe("fami_cuartoshogar",        str(ui["fami_cuartoshogar"]))
        _set_ohe("fami_educacionmadre",      str(ui["fami_educacionmadre"]))
        _set_ohe("fami_educacionpadre",      str(ui["fami_educacionpadre"]))
        _set_ohe("fami_estratovivienda",     str(ui["fami_estratovivienda"]))
        _set_ohe("fami_personashogar",       str(ui["fami_personashogar"]))

        estab = ui.get("cole_cod_dane_establecimiento")
        if estab:
            _set_ohe("cole_cod_dane_establecimiento", str(int(estab)))

        series = pd.Series(raw)
        series = series.reindex(self.feature_columns, fill_value=0)
        return series
