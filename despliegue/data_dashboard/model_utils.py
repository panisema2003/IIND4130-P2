"""
model_utils.py
Utilidades de preprocesamiento para Q1 — Predictor de puntaje ICFES.
Refleja el pipeline de question1.ipynb para predicciones sobre una sola fila.
Cargado una vez al iniciar la app.
"""

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Tabla de colegios cargada desde model/colegios.json (generada por build_colegios.py).
# Clave: str(codigo DANE), Valor: {nombre, municipio, area_urbano, bilingue, ...}

def _cargar_colegios() -> dict:
    path = os.path.join(os.path.dirname(__file__), "model", "colegios.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

COLEGIOS: dict = _cargar_colegios()

# Agrupados por municipio para filtrar el desplegable de colegios
COLEGIOS_BY_MCPIO: dict[int, list[dict]] = {}
for _code, _info in COLEGIOS.items():
    _m = _info["municipio"]
    COLEGIOS_BY_MCPIO.setdefault(_m, []).append({"code": _code, **_info})

# Municipios del departamento de Cesar
MUNICIPIOS_CESAR = {
    20001: "Valledupar",
    20011: "Aguachica",
    20013: "Agustin Codazzi",
    20032: "Astrea",
    20045: "Becerril",
    20060: "Bosconia",
    20175: "Chimichagua",
    20178: "Chiriguana",
    20228: "Curumani",
    20238: "El Copey",
    20250: "El Paso",
    20295: "Gamarra",
    20310: "Gonzalez",
    20383: "La Gloria",
    20400: "La Jagua de Ibirico",
    20443: "Manaure",
    20517: "Pailitas",
    20550: "Pelaya",
    20570: "Pueblo Bello",
    20614: "Rio de Oro",
    20621: "La Paz",
    20710: "San Alberto",
    20750: "San Diego",
    20770: "San Martin",
    20787: "Tamalameque",
}

# Etiquetas legibles para niveles de educacion
EDUCATION_LABELS = {
    "Ninguno":                                  "Ninguno",
    "No Aplica":                                "No aplica",
    "No sabe":                                  "No sabe",
    "Primaria incompleta":                      "Primaria incompleta",
    "Primaria completa":                        "Primaria completa",
    "Secundaria (Bachillerato) incompleta":     "Bachillerato incompleto",
    "Secundaria (Bachillerato) completa":       "Bachillerato completo",
    "Tecnica o tecnologica incompleta":         "Tecnica/Tecnologica incompleta",
    "Tecnica o tecnologica completa":           "Tecnica/Tecnologica completa",
    "Educacion profesional incompleta":         "Profesional incompleto",
    "Educacion profesional completa":           "Profesional completo",
    "Postgrado":                                "Postgrado",
}

PERIODOS_LABELS = {
    20142: "2014-2",
    20151: "2015-1",
    20152: "2015-2",
    20161: "2016-1",
    20162: "2016-2",
    20171: "2017-1",
    20172: "2017-2",
    20181: "2018-1",
    20191: "2019-1",
    20194: "2019-4",
    20201: "2020-1",
    20211: "2021-1",
    20221: "2022-1",
    20224: "2022-4",
}

CUARTOS_MAP = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6+": "6+",
}

PERSONAS_MAP = {
    "1 a 2": "1 a 2",
    "3 a 4": "3 a 4",
    "5 a 6": "5 a 6",
    "7 a 8": "7 a 8",
    "9 o mas": "9 o mas",
    "12 o mas": "12 o mas",
}

# Grupos de variables para la explicacion por ablacion
# Cada entrada: (etiqueta, definicion_grupo)
# definicion_grupo = lista de nombres exactos  O  prefijo str para coincidir todas las columnas

FEATURE_GROUPS = [
    ("Tipo de colegio (oficial/privado)",   ["cole_oficial"]),
    ("Colegio bilingue",                    ["cole_bilingue"]),
    ("Area del colegio (urbano/rural)",     ["cole_area_urbano"]),
    ("Calendario escolar",                  ["cole_calendario_a"]),
    ("Tecnologia en el hogar",              ["fami_tienecomputador", "fami_tieneinternet",
                                             "fami_tieneautomovil", "fami_tienelavadora"]),
    ("Educacion de la madre",               "fami_educacionmadre_"),
    ("Educacion del padre",                 "fami_educacionpadre_"),
    ("Estrato socioeconomico",              "fami_estratovivienda_"),
    ("Tamano del hogar",                    "fami_personashogar_"),
    ("Espacio en el hogar (cuartos)",       "fami_cuartoshogar_"),
    ("Jornada escolar",                     "cole_jornada_"),
    ("Caracter del colegio",               "cole_caracter_"),
    ("Municipio del colegio",               "cole_cod_mcpio_ubicacion_"),
    ("Municipio de residencia",             "estu_cod_reside_mcpio_"),
    ("Genero del estudiante",               ["estu_masculino"]),
    ("Nacionalidad",                        "estu_nacionalidad_"),
    ("Establecimiento educativo",           "cole_cod_dane_establecimiento_"),
    ("Periodo del examen",                  ["periodo"]),
]


class ModelLoader:
    """Carga y almacena en cache el modelo Q1 y sus artefactos al iniciar la app."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns: list[str] = []
        self.model_info: dict = {}
        self.background_mean: np.ndarray | None = None
        self.shap_global: list = []
        self._load()

    def _load(self):
        model_path = os.path.join(self.model_dir, "model.keras")
        feat_path  = os.path.join(self.model_dir, "feature_columns.json")
        info_path  = os.path.join(self.model_dir, "model_info.json")
        bg_path    = os.path.join(self.model_dir, "background_mean.npy")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}.\n"
                "Ejecutar  python export_model.py  para entrenar y guardar el modelo."
            )

        # Keras 3.x elimino los params de renorm en BatchNormalization; se usa wrapper de compatibilidad
        class _CompatBN(tf.keras.layers.BatchNormalization):
            def __init__(self, **kwargs):
                for k in ("renorm", "renorm_clipping", "renorm_momentum"):
                    kwargs.pop(k, None)
                super().__init__(**kwargs)

        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"BatchNormalization": _CompatBN}
        )
        with open(feat_path) as f:
            self.feature_columns = json.load(f)
        with open(info_path) as f:
            self.model_info = json.load(f)
        if os.path.exists(bg_path):
            self.background_mean = np.load(bg_path)

        shap_path = os.path.join(self.model_dir, "shap_global.json")
        if os.path.exists(shap_path):
            with open(shap_path, encoding="utf-8") as f:
                self.shap_global: list = json.load(f)
        else:
            self.shap_global = []

    def predict(self, user_inputs: dict) -> float:
        """
        Construye una fila preprocesada a partir de user_inputs y retorna la prediccion.

        Entradas requeridas (todas):
            periodo              int   ej. 20221
            cole_area_urbano     bool  True=Urbano, False=Rural
            cole_bilingue        bool
            cole_calendario_a    bool  True=A, False=B
            cole_oficial         bool  True=Oficial, False=No oficial
            estu_masculino       bool  True=Masculino, False=Femenino
            fami_tieneautomovil  bool
            fami_tienecomputador bool
            fami_tieneinternet   bool
            fami_tienelavadora   bool
            cole_caracter        str   ej. "ACADEMICO"
            cole_jornada         str   ej. "MANANA"
            estu_nacionalidad    str   ej. "COLOMBIA"
            fami_cuartoshogar    str   ej. "3"
            fami_educacionmadre  str   etiqueta de nivel educativo
            fami_educacionpadre  str   etiqueta de nivel educativo
            fami_estratovivienda str   ej. "Estrato 2"
            fami_personashogar   str   ej. "3 a 4"
            cole_cod_mcpio_ubicacion  int   codigo DANE del municipio
            estu_cod_reside_mcpio     int   codigo DANE del municipio
            cole_cod_dane_establecimiento  int  (opcional, 0 si no se conoce)
            cole_cod_dane_sede            int  (opcional, 0 si no se conoce)
        """
        row = self._build_feature_row(user_inputs)
        x = row.to_numpy(dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(x, verbose=0)[0][0]
        return float(pred)

    def explain_groups(self, user_inputs: dict, top_n: int = 5) -> list:
        """
        Ablacion por grupos: reemplaza cada grupo con la media del entrenamiento
        y mide el cambio en la prediccion.

        impacto = pred_baseline - pred_ablada
          > 0 -> el grupo contribuye positivamente al puntaje del estudiante
          < 0 -> el grupo reduce el puntaje respecto al promedio
        """
        if self.background_mean is None:
            return []

        baseline_row = self._build_feature_row(user_inputs).to_numpy(dtype=np.float32)
        feat_idx = {c: i for i, c in enumerate(self.feature_columns)}

        ablated_rows = []
        group_labels = []

        for label, group_def in FEATURE_GROUPS:
            row = baseline_row.copy()
            if isinstance(group_def, list):
                for feat_name in group_def:
                    if feat_name in feat_idx:
                        i = feat_idx[feat_name]
                        row[i] = self.background_mean[i]
            else:
                prefix = group_def
                for feat_name in self.feature_columns:
                    if feat_name.startswith(prefix):
                        i = feat_idx[feat_name]
                        row[i] = self.background_mean[i]
            ablated_rows.append(row)
            group_labels.append(label)

        batch = np.array(ablated_rows, dtype=np.float32)
        baseline_pred = float(self.model.predict(baseline_row.reshape(1, -1), verbose=0)[0][0])
        ablated_preds = self.model.predict(batch, verbose=0).flatten()

        impacts = [
            (lbl, float(baseline_pred - abl_pred))
            for lbl, abl_pred in zip(group_labels, ablated_preds)
        ]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return [{"label": lbl, "impact": imp} for lbl, imp in impacts[:top_n]]

    def _build_feature_row(self, ui: dict) -> pd.Series:
        """Reconstruye una fila preprocesada alineada a feature_columns."""

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
            # Pone 1 en la columna OHE correspondiente; reindex descarta las desconocidas
            col = f"{prefix}_{value}"
            raw[col] = 1

        _set_ohe("cole_caracter",               str(ui["cole_caracter"]))
        _set_ohe("cole_cod_mcpio_ubicacion",     str(int(ui["cole_cod_mcpio_ubicacion"])))
        _set_ohe("estu_cod_reside_mcpio",        str(int(ui["estu_cod_reside_mcpio"])))
        _set_ohe("cole_jornada",                 str(ui["cole_jornada"]))
        _set_ohe("estu_nacionalidad",            str(ui["estu_nacionalidad"]))
        _set_ohe("fami_cuartoshogar",            str(ui["fami_cuartoshogar"]))
        _set_ohe("fami_educacionmadre",          str(ui["fami_educacionmadre"]))
        _set_ohe("fami_educacionpadre",          str(ui["fami_educacionpadre"]))
        _set_ohe("fami_estratovivienda",         str(ui["fami_estratovivienda"]))
        _set_ohe("fami_personashogar",           str(ui["fami_personashogar"]))

        estab = ui.get("cole_cod_dane_establecimiento")
        if estab:
            _set_ohe("cole_cod_dane_establecimiento", str(int(estab)))

        sede = ui.get("cole_cod_dane_sede")
        if sede:
            _set_ohe("cole_cod_dane_sede", str(int(sede)))

        series = pd.Series(raw)
        series = series.reindex(self.feature_columns, fill_value=0)
        return series
