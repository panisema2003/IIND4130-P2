"""
model_utils_q2.py
Utilidades de preprocesamiento para Q2 — Clasificador de colegios bilingues.
Refleja el pipeline de data_science_2/export_model_q2.py.
"""

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Opciones para los desplegables del formulario Q2

NATURALEZA_OPTIONS = [
    {"label": "Oficial (Publica)",    "value": "OFICIAL"},
    {"label": "No Oficial (Privada)", "value": "NO OFICIAL"},
]

JORNADA_OPTIONS = [
    {"label": "Unica",    "value": "UNICA"},
    {"label": "Manana",   "value": "MANANA"},
    {"label": "Tarde",    "value": "TARDE"},
    {"label": "Completa", "value": "COMPLETA"},
    {"label": "Sabatina", "value": "SABATINA"},
    {"label": "Noche",    "value": "NOCHE"},
]

GENERO_OPTIONS = [
    {"label": "Femenino",  "value": "F"},
    {"label": "Masculino", "value": "M"},
]

INTERNET_OPTIONS = [
    {"label": "No", "value": "No"},
    {"label": "Si", "value": "Si"},
]

COMPUTADOR_OPTIONS = [
    {"label": "No", "value": "No"},
    {"label": "Si", "value": "Si"},
]

ESTRATO_OPTIONS_Q2 = [
    {"label": "Sin Estrato / No aplica", "value": 2},
    {"label": "Estrato 1", "value": 1},
    {"label": "Estrato 2", "value": 2},
    {"label": "Estrato 3", "value": 3},
    {"label": "Estrato 4", "value": 4},
    {"label": "Estrato 5", "value": 5},
    {"label": "Estrato 6", "value": 6},
]

# Grupos de variables para la explicacion por ablacion
FEATURE_GROUPS_Q2 = [
    ("Puntaje Ingles",             ["punt_ingles"]),
    ("Puntaje Global",             ["punt_global"]),
    ("Puntaje Lectura Critica",    ["punt_lectura_critica"]),
    ("Puntaje Matematicas",        ["punt_matematicas"]),
    ("Puntaje Ciencias Naturales", ["punt_c_naturales"]),
    ("Puntaje Sociales",           ["punt_sociales_ciudadanas"]),
    ("Estrato Socioeconomico",     ["fami_estratovivienda"]),
    ("Tipo de Colegio",            "cole_naturaleza_"),
    ("Jornada Escolar",            "cole_jornada_"),
    ("Genero del Estudiante",      "estu_genero_"),
    ("Internet en el Hogar",       "fami_tieneinternet_"),
    ("Computador en el Hogar",     "fami_tienecomputador_"),
]


class ModelLoaderQ2:
    """Carga y almacena en cache el modelo Q2 y sus artefactos al iniciar la app."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns: list[str] = []
        self.model_info: dict = {}
        self.class_weights: dict = {}
        self.threshold: float = 0.5
        self.background_mean: np.ndarray | None = None
        self._available = False
        self._load()

    def _load(self):
        model_path = os.path.join(self.model_dir, "model.keras")
        if not os.path.exists(model_path):
            return

        feat_path = os.path.join(self.model_dir, "feature_columns.json")
        info_path = os.path.join(self.model_dir, "model_info.json")
        cw_path   = os.path.join(self.model_dir, "class_weights.json")
        thr_path  = os.path.join(self.model_dir, "threshold.json")
        bg_path   = os.path.join(self.model_dir, "background_mean.npy")

        self.model = tf.keras.models.load_model(model_path)

        with open(feat_path, encoding="utf-8") as f:
            self.feature_columns = json.load(f)
        with open(info_path, encoding="utf-8") as f:
            self.model_info = json.load(f)
        with open(cw_path) as f:
            cw = json.load(f)
            self.class_weights = {int(k): float(v) for k, v in cw.items()}
        with open(thr_path) as f:
            self.threshold = float(json.load(f).get("threshold", 0.5))
        if os.path.exists(bg_path):
            self.background_mean = np.load(bg_path)

        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, user_inputs: dict) -> dict:
        """
        Retorna probabilidad y etiqueta de clasificacion.

        Entradas requeridas:
            punt_lectura_critica, punt_matematicas, punt_sociales_ciudadanas,
            punt_c_naturales, punt_ingles, punt_global  -> float
            fami_estratovivienda -> int 1-6
            cole_naturaleza  -> "OFICIAL" | "NO OFICIAL"
            cole_jornada     -> "UNICA" | "MANANA" | "TARDE" | "COMPLETA" | "SABATINA" | "NOCHE"
            estu_genero      -> "F" | "M"
            fami_tieneinternet   -> "Si" | "No"
            fami_tienecomputador -> "Si" | "No"
        """
        row = self._build_feature_row(user_inputs)
        x = row.to_numpy(dtype=np.float32).reshape(1, -1)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        es_bilingue = prob >= self.threshold
        return {
            "probability":  prob,
            "label":        "Bilingue" if es_bilingue else "No Bilingue",
            "is_bilingual": es_bilingue,
        }

    def explain_groups(self, user_inputs: dict, top_n: int = 5) -> list:
        """
        Ablacion por grupos: reemplaza cada grupo de variables con la media del
        entrenamiento y mide el cambio en probabilidad predicha.

        impacto > 0 -> el grupo contribuye positivamente a la clasificacion bilingue
        impacto < 0 -> el grupo reduce la probabilidad de clasificacion bilingue
        """
        if self.background_mean is None:
            return []

        baseline = self._build_feature_row(user_inputs).to_numpy(dtype=np.float32)
        feat_idx = {c: i for i, c in enumerate(self.feature_columns)}

        ablated_rows, labels = [], []
        for label, group_def in FEATURE_GROUPS_Q2:
            row = baseline.copy()
            if isinstance(group_def, list):
                for name in group_def:
                    if name in feat_idx:
                        row[feat_idx[name]] = self.background_mean[feat_idx[name]]
            else:
                prefix = group_def
                for name in self.feature_columns:
                    if name.startswith(prefix):
                        row[feat_idx[name]] = self.background_mean[feat_idx[name]]
            ablated_rows.append(row)
            labels.append(label)

        batch = np.array(ablated_rows, dtype=np.float32)
        base_prob = float(self.model.predict(baseline.reshape(1, -1), verbose=0)[0][0])
        abl_probs = self.model.predict(batch, verbose=0).flatten()

        impacts = [
            (lbl, float(base_prob - abl_p))
            for lbl, abl_p in zip(labels, abl_probs)
        ]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return [{"label": lbl, "impact": imp} for lbl, imp in impacts[:top_n]]

    def _build_feature_row(self, ui: dict) -> pd.Series:
        """Construye una fila de variables alineada a feature_columns."""
        raw: dict = {}

        for feat in [
            "punt_lectura_critica", "punt_matematicas", "punt_sociales_ciudadanas",
            "punt_c_naturales", "punt_ingles", "punt_global",
        ]:
            raw[feat] = float(ui.get(feat) or 0)

        estrato = ui.get("fami_estratovivienda")
        raw["fami_estratovivienda"] = float(estrato) if estrato is not None else 2.0

        # OHE: pone 1 en la columna correspondiente; reindex elimina las desconocidas
        for cat in ["cole_naturaleza", "cole_jornada", "estu_genero",
                    "fami_tieneinternet", "fami_tienecomputador"]:
            val = ui.get(cat, "")
            if val:
                raw[f"{cat}_{val}"] = 1.0

        series = pd.Series(raw)
        series = series.reindex(self.feature_columns, fill_value=0.0)
        return series
