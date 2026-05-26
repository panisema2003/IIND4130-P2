"""
app.py — Dashboard Ministerio de Educación
Prueba Saber 11, Departamento del Cesar

Q1: Predicción de puntaje global (regresión)
Q2: Clasificación de colegios bilingues
"""

import os
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from model_utils import (
    ModelLoader, MUNICIPIOS_CESAR, PERIODOS_LABELS, EDUCATION_LABELS,
    COLEGIOS, COLEGIOS_BY_MCPIO,
)
from model_utils_q2 import (
    ModelLoaderQ2,
    NATURALEZA_OPTIONS, JORNADA_OPTIONS, GENERO_OPTIONS,
    INTERNET_OPTIONS, COMPUTADOR_OPTIONS, ESTRATO_OPTIONS_Q2,
)
from model_utils_q3 import (
    ModelLoaderQ3,
    TARGET_COLS as Q3_TARGET_COLS,
    TARGET_LABELS as Q3_TARGET_LABELS,
    TARGET_COLORS as Q3_TARGET_COLORS,
    CUARTOS_OPTIONS_Q3, PERSONAS_OPTIONS_Q3,
)

VINOTINTO = "#7B1C2B"
CREMA     = "#FAF5EC"
CREMA2    = "#F0E8D5"

MODEL_DIR    = "model"
MODEL_DIR_Q2 = "model_q2"
MODEL_DIR_Q3 = "model_q3"

loader    = ModelLoader(MODEL_DIR)
loader_q2 = ModelLoaderQ2(MODEL_DIR_Q2)
loader_q3 = ModelLoaderQ3(MODEL_DIR_Q3)
Q2_AVAILABLE = loader_q2.available
Q3_AVAILABLE = loader_q3.available
info    = loader.model_info
info_q2 = loader_q2.model_info if Q2_AVAILABLE else {}
info_q3 = loader_q3.model_info if Q3_AVAILABLE else {}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    title="Predictor ICFES - Ministerio de Educación",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server


# --- Funciones auxiliares Q1 ---

def nivel_puntaje(score: float):
    if score < 200:   return "Bajo",       "#c0392b"
    elif score < 250: return "Medio-Bajo", "#d35400"
    elif score < 300: return "Medio",      "#b7950b"
    elif score < 350: return "Medio-Alto", "#1e8449"
    else:             return "Alto",       "#1a5276"


def grafico_gauge_q1(score: float) -> go.Figure:
    label, color = nivel_puntaje(score)
    mae = info["metrics"]["mae"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 1),
        number={"font": {"size": 52, "color": color}, "suffix": " pts"},
        delta={
            "reference": info["target_mean"], "valueformat": ".1f",
            "increasing": {"color": "#1e8449"}, "decreasing": {"color": "#c0392b"},
        },
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#888",
                     "tickvals": [0, 100, 200, 250, 300, 350, 400, 500]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white", "borderwidth": 2, "bordercolor": CREMA2,
            "steps": [
                {"range": [0,   200], "color": "#fde8e8"},
                {"range": [200, 250], "color": "#fef3e2"},
                {"range": [250, 300], "color": "#fefbe6"},
                {"range": [300, 350], "color": "#e8f8f0"},
                {"range": [350, 500], "color": "#d6eaf8"},
            ],
            "threshold": {"line": {"color": "#555", "width": 3},
                          "thickness": 0.8, "value": info["target_mean"]},
        },
        title={"text": f"Puntaje Predicho<br>"
                       f"<span style='font-size:0.9em;color:{color}'>{label}</span>",
               "font": {"size": 18}},
    ))
    fig.add_annotation(
        text=f"Rango probable: {max(0, score - mae):.0f} – {min(500, score + mae):.0f} pts",
        x=0.5, y=-0.08, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color="#888"),
    )
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=55), height=300,
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


def grafico_shap_global() -> go.Figure:
    grupos = loader.shap_global
    if not grupos:
        return go.Figure()
    labels = [g["label"] for g in reversed(grupos)]
    values = [g["importance"] for g in reversed(grupos)]
    max_v  = max(values) if values else 1
    colores = [f"rgba(123,28,43,{0.35 + 0.65*(v/max_v):.2f})" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colores,
        text=[f"{v:.1f} pts" for v in values], textposition="outside",
        textfont=dict(size=11), hovertemplate="%{y}: %{x:.2f} pts<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Media |SHAP| (pts)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=30),
        height=max(280, len(grupos) * 28 + 60),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def grafico_barra_contexto(score: float) -> go.Figure:
    mean = info["target_mean"]
    mae  = info["metrics"]["mae"]
    fig  = go.Figure()
    bandas = [
        (0,   200, "#fde8e8", "Bajo"),
        (200, 250, "#fef3e2", "Medio-Bajo"),
        (250, 300, "#fefbe6", "Medio"),
        (300, 350, "#e8f8f0", "Medio-Alto"),
        (350, 500, "#d6eaf8", "Alto"),
    ]
    for lo, hi, clr, lbl in bandas:
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1,
                      fillcolor=clr, line_width=0, layer="below")
        fig.add_annotation(x=(lo+hi)/2, y=0.5, text=lbl, showarrow=False,
                           font=dict(size=10, color="#888"), yref="paper")
    fig.add_vline(x=mean, line_dash="dash", line_color="#aaa", line_width=1.5,
                  annotation_text=f"Promedio hist. ({mean:.0f})",
                  annotation_position="top left", annotation_font_size=11,
                  annotation_yshift=12)
    fig.add_shape(type="rect", x0=max(0, score-mae), x1=min(500, score+mae),
                  y0=0.2, y1=0.8, fillcolor="rgba(123,28,43,0.18)",
                  line=dict(color=VINOTINTO, width=1.5), layer="above")
    fig.add_scatter(x=[score], y=[0.5], mode="markers",
                    marker=dict(size=18, color=VINOTINTO, symbol="diamond"),
                    showlegend=False)
    fig.update_layout(
        xaxis=dict(range=[0, 500], title="Puntaje Global",
                   tickvals=[0, 100, 200, 250, 300, 350, 400, 500]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=110, margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


# --- Funciones auxiliares Q2 ---

def grafico_gauge_q2(prob: float, es_bilingue: bool) -> go.Figure:
    pct   = round(prob * 100, 1)
    color = "#1e8449" if es_bilingue else "#c0392b"
    label = "BILINGÜE" if es_bilingue else "NO BILINGÜE"
    umbral_pct = loader_q2.threshold * 100 if Q2_AVAILABLE else 50
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"font": {"size": 44, "color": color}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888",
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white", "borderwidth": 2, "bordercolor": CREMA2,
            "steps": [
                {"range": [0,  50],  "color": "#fde8e8"},
                {"range": [50, 100], "color": "#e8f8f0"},
            ],
            "threshold": {"line": {"color": "#555", "width": 3},
                          "thickness": 0.8, "value": umbral_pct},
        },
        title={"text": f"Probabilidad de ser Bilingüe<br>"
                       f"<span style='font-size:0.9em;color:{color}'>{label}</span>",
               "font": {"size": 16}},
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=20), height=280,
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


def badge_bilingue(es_bilingue: bool):
    if es_bilingue:
        return dbc.Badge(
            [html.I(className="bi bi-check-circle-fill me-1"), "BILINGÜE"],
            color="success", className="fs-5 px-3 py-2",
        )
    return dbc.Badge(
        [html.I(className="bi bi-x-circle-fill me-1"), "NO BILINGÜE"],
        color="danger", className="fs-5 px-3 py-2",
    )


# --- Funciones auxiliares Q3 ---

def grafico_radar_q3(scores: dict) -> go.Figure:
    per_subject = info_q3["metrics"]["per_subject"]
    labels = Q3_TARGET_LABELS
    vals   = [scores[c] for c in Q3_TARGET_COLS]
    means  = [per_subject[c]["mean"] for c in Q3_TARGET_COLS]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=means + [means[0]], theta=labels + [labels[0]],
        fill="toself", name="Promedio histórico Cesar",
        line=dict(color="#aaaaaa", dash="dash", width=1.5),
        fillcolor="rgba(180,180,180,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", name="Predicción",
        line=dict(color=VINOTINTO, width=2.5),
        fillcolor="rgba(123,28,43,0.18)",
        marker=dict(size=8, color=VINOTINTO),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                tickfont=dict(size=10), gridcolor="#e8e8e8",
            ),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    font=dict(size=11)),
        height=380,
        margin=dict(l=50, r=50, t=30, b=60),
        paper_bgcolor="white",
    )
    return fig


def tarjetas_materias(scores: dict) -> list:
    per_subject = info_q3["metrics"]["per_subject"]
    cards = []
    for col, label in zip(Q3_TARGET_COLS, Q3_TARGET_LABELS):
        score = scores[col]
        meta  = per_subject[col]
        color = Q3_TARGET_COLORS[col]
        delta = score - meta["mean"]
        delta_txt = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        delta_col = "#1e8449" if delta >= 0 else "#c0392b"
        cards.append(dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div(label, className="small fw-semibold text-center mb-1",
                         style={"color": color, "fontSize": "0.75rem"}),
                html.Div(f"{score:.1f}", className="text-center fw-bold",
                         style={"fontSize": "1.5rem", "color": color, "lineHeight": "1"}),
                html.Div("pts", className="text-center text-muted",
                         style={"fontSize": "0.7rem"}),
                html.Div(delta_txt, className="text-center small fw-semibold mt-1",
                         style={"color": delta_col, "fontSize": "0.78rem"}),
            ], className="py-2 px-1"),
        ], className="shadow-sm h-100", style={"borderTop": f"3px solid {color}"})))
    return cards


# --- Componentes de layout reutilizables ---

def campo(label: str, componente):
    return html.Div([
        html.Label(label, className="form-label fw-semibold mb-1 input-label"),
        componente,
    ], className="mb-3")


def tarjeta_seccion(titulo: str, icono: str, contenido):
    return dbc.Card([
        dbc.CardHeader(
            html.H6([html.I(className=f"bi {icono} me-2"), titulo],
                    className="mb-0 fw-bold section-title"),
            className="section-header",
        ),
        dbc.CardBody(contenido, className="py-3 px-3"),
    ], className="mb-3 shadow-sm")


# --- Opciones de dropdowns compartidos ---

opciones_mcpio = [
    {"label": f"{nombre} ({codigo})", "value": codigo}
    for codigo, nombre in sorted(MUNICIPIOS_CESAR.items(), key=lambda x: x[1])
]
opciones_periodo = [
    {"label": label, "value": codigo}
    for codigo, label in sorted(PERIODOS_LABELS.items())
]
opciones_educacion = [
    {"label": label, "value": raw}
    for raw, label in EDUCATION_LABELS.items()
]


# --- Layout Q1 ---

def layout_q1():
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Prediccion de puntaje global: "),
            "Con base en el contexto del estudiante, su familia y su colegio, "
            "el modelo estima el ",
            html.Strong("Puntaje Global"),
            " esperado en la Prueba Saber 11 para el Departamento del Cesar.",
            html.Br(),
            html.Small([
                f"Modelo: arch_D_deep — {info['n_features']} variables — "
                f"MAE = {info['metrics']['mae']:.1f} pts — "
                f"R² = {info['metrics']['r2']:.3f} — "
                f"N entrenamiento = {info['n_train']:,}",
            ], className="text-muted"),
        ], color="light", dismissable=False, className="mb-3 py-2 banner-alert"),

        dbc.Row([
            dbc.Col([
                tarjeta_seccion("Periodo del Examen", "bi-calendar3", [
                    campo("Periodo", dcc.Dropdown(
                        id="in-periodo", options=opciones_periodo,
                        value=20221, clearable=False,
                    )),
                ]),

                tarjeta_seccion("Datos del Estudiante", "bi-person", [
                    dbc.Row([
                        dbc.Col(campo("Genero", dcc.Dropdown(
                            id="in-genero",
                            options=[{"label": "Masculino", "value": True},
                                     {"label": "Femenino",  "value": False}],
                            value=True, clearable=False,
                        )), md=6),
                        dbc.Col(campo("Nacionalidad", dcc.Dropdown(
                            id="in-nacionalidad",
                            options=[
                                {"label": "Colombia",  "value": "COLOMBIA"},
                                {"label": "Venezuela", "value": "VENEZUELA"},
                                {"label": "Ecuador",   "value": "ECUADOR"},
                                {"label": "España",    "value": "ESPAÑA"},
                            ],
                            value="COLOMBIA", clearable=False,
                        )), md=6),
                    ]),
                    campo("Municipio de Residencia", dcc.Dropdown(
                        id="in-mcpio-residencia", options=opciones_mcpio,
                        value=20001, clearable=False,
                    )),
                ]),

                tarjeta_seccion("Contexto Familiar", "bi-house", [
                    dbc.Row([
                        dbc.Col(campo("Estrato de Vivienda", dcc.Dropdown(
                            id="in-estrato",
                            options=[
                                {"label": "Sin Estrato", "value": "Sin Estrato"},
                                {"label": "Estrato 1",   "value": "Estrato 1"},
                                {"label": "Estrato 2",   "value": "Estrato 2"},
                                {"label": "Estrato 3",   "value": "Estrato 3"},
                                {"label": "Estrato 4",   "value": "Estrato 4"},
                                {"label": "Estrato 5",   "value": "Estrato 5"},
                                {"label": "Estrato 6",   "value": "Estrato 6"},
                            ],
                            value="Estrato 1", clearable=False,
                        )), md=6),
                        dbc.Col(campo("Personas en el Hogar", dcc.Dropdown(
                            id="in-personas",
                            options=[
                                {"label": "1 a 2",    "value": "1 a 2"},
                                {"label": "3 a 4",    "value": "3 a 4"},
                                {"label": "5 a 6",    "value": "5 a 6"},
                                {"label": "7 a 8",    "value": "7 a 8"},
                                {"label": "9 o mas",  "value": "9 o más"},
                                {"label": "12 o mas", "value": "12 o más"},
                            ],
                            value="3 a 4", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Cuartos en el Hogar", dcc.Dropdown(
                            id="in-cuartos",
                            options=[
                                {"label": "1 cuarto",  "value": "1"},
                                {"label": "2 cuartos", "value": "2"},
                                {"label": "3 cuartos", "value": "3"},
                                {"label": "4 cuartos", "value": "4"},
                                {"label": "5 cuartos", "value": "5"},
                                {"label": "6 a 9",     "value": "6+"},
                                {"label": "10 o mas",  "value": "10+"},
                            ],
                            value="3", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Educacion de la Madre",
                            dcc.Dropdown(id="in-edu-madre", options=opciones_educacion,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                        dbc.Col(campo("Educacion del Padre",
                            dcc.Dropdown(id="in-edu-padre", options=opciones_educacion,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                    ]),
                    html.Label("Recursos del hogar",
                               className="form-label fw-semibold mb-2 input-label"),
                    dbc.Checklist(
                        id="in-recursos",
                        options=[
                            {"label": "Computador", "value": "computador"},
                            {"label": "Internet",   "value": "internet"},
                            {"label": "Automovil",  "value": "automovil"},
                            {"label": "Lavadora",   "value": "lavadora"},
                        ],
                        value=["computador", "internet"],
                        switch=True, inline=False,
                    ),
                ]),

                tarjeta_seccion("Contexto Escolar", "bi-building", [
                    campo("Municipio del Colegio", dcc.Dropdown(
                        id="in-mcpio-cole", options=opciones_mcpio,
                        value=20001, clearable=False,
                    )),
                    campo("Colegio", dcc.Dropdown(
                        id="in-colegio", options=[],
                        placeholder="Seleccione el municipio primero...", clearable=True,
                    )),
                    dbc.Alert(
                        [html.I(className="bi bi-magic me-2"),
                         "Al seleccionar el colegio los campos de abajo se completan "
                         "automaticamente. Puede ajustarlos manualmente."],
                        color="light", className="py-1 px-2 mb-3 small banner-alert",
                        dismissable=False,
                    ),
                    dbc.Row([
                        dbc.Col(campo("Area", dcc.Dropdown(
                            id="in-area",
                            options=[{"label": "Urbano", "value": True},
                                     {"label": "Rural",  "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(campo("Calendario", dcc.Dropdown(
                            id="in-calendario",
                            options=[{"label": "Calendario A", "value": True},
                                     {"label": "Calendario B", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(campo("Naturaleza", dcc.Dropdown(
                            id="in-naturaleza",
                            options=[{"label": "Oficial (Publica)",   "value": True},
                                     {"label": "No Oficial (Privada)", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Jornada", dcc.Dropdown(
                            id="in-jornada",
                            options=[
                                {"label": "Unica",    "value": "UNICA"},
                                {"label": "Manana",   "value": "MAÑANA"},
                                {"label": "Tarde",    "value": "TARDE"},
                                {"label": "Completa", "value": "COMPLETA"},
                                {"label": "Sabatina", "value": "SABATINA"},
                                {"label": "Noche",    "value": "NOCHE"},
                            ],
                            value="UNICA", clearable=False,
                        )), md=4),
                        dbc.Col(campo("Caracter", dcc.Dropdown(
                            id="in-caracter",
                            options=[
                                {"label": "Academico",           "value": "ACADÉMICO"},
                                {"label": "Tecnico",             "value": "TÉCNICO"},
                                {"label": "Tecnico / Academico", "value": "TÉCNICO/ACADÉMICO"},
                            ],
                            value="ACADÉMICO", clearable=False,
                        )), md=4),
                        dbc.Col(campo("Bilingue", dcc.Dropdown(
                            id="in-bilingue",
                            options=[{"label": "No", "value": False},
                                     {"label": "Si", "value": True}],
                            value=False, clearable=False,
                        )), md=4),
                    ]),
                ]),

                dbc.Button(
                    [html.I(className="bi bi-graph-up-arrow me-2"), "Predecir Puntaje"],
                    id="btn-predict", size="lg",
                    className="w-100 mb-2 fw-bold predict-btn",
                ),
                html.Small(
                    "La prediccion es una estimacion basada en datos historicos del Cesar.",
                    className="text-muted d-block text-center mb-4",
                ),
            ], md=5),

            dbc.Col([
                html.Div(id="result-panel", children=[
                    html.Div([
                        html.Div(html.I(className="bi bi-clipboard-data",
                                        style={"fontSize": "3.5rem", "color": CREMA2}),
                                 className="text-center mb-3"),
                        html.H5("Complete el formulario y presione Predecir",
                                className="text-center text-muted"),
                        html.P("El modelo estimara el puntaje global esperado.",
                               className="text-center text-muted small"),
                    ], className="py-5 px-3"),
                ]),

                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-info-circle me-2"),
                         "Como interpretar el resultado"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        html.P("El Puntaje Global Saber 11 va de 0 a 500 puntos:",
                               className="mb-2 small"),
                        html.Div([
                            html.Div([
                                html.Span(style={"display": "inline-block", "width": "12px",
                                                 "height": "12px", "borderRadius": "2px",
                                                 "backgroundColor": bg, "marginRight": "6px",
                                                 "verticalAlign": "middle"}),
                                html.Span(f"{lo}-{hi}: {lbl}", className="small"),
                            ], className="mb-1")
                            for lo, hi, lbl, bg in [
                                ("0",   "200", "Bajo",       "#f1948a"),
                                ("200", "250", "Medio-Bajo", "#f0b27a"),
                                ("250", "300", "Medio",      "#f9e79f"),
                                ("300", "350", "Medio-Alto", "#82e0aa"),
                                ("350", "500", "Alto",       "#7fb3d3"),
                            ]
                        ]),
                        html.Hr(className="my-2"),
                        html.P([html.Strong("Rango de confianza: "),
                                f"+-{info['metrics']['mae']:.0f} pts (MAE del modelo)."],
                               className="small mb-1"),
                        html.P([html.Strong("Referencia historica: "),
                                f"Promedio Cesar = {info['target_mean']:.0f} pts."],
                               className="small mb-0"),
                    ], className="py-2 px-3"),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-cpu me-2"), "Metricas del Modelo Q1"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.Div(f"{info['metrics']['mae']:.1f}",
                                              className="metric-value", style={"color": "#c0392b"}),
                                     html.Div("MAE (pts)", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{info['metrics']['rmse']:.1f}",
                                              className="metric-value", style={"color": "#d35400"}),
                                     html.Div("RMSE (pts)", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{info['metrics']['r2']:.3f}",
                                              className="metric-value", style={"color": "#1e8449"}),
                                     html.Div("R2", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{info['n_features']}",
                                              className="metric-value", style={"color": VINOTINTO}),
                                     html.Div("Variables", className="metric-label")],
                                    className="text-center"),
                        ]),
                    ], className="py-2"),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-bar-chart-line me-2"),
                         "Importancia Global de Variables (SHAP)"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        html.P(
                            "Media del valor absoluto SHAP por grupo de variables, "
                            "calculada sobre 300 estudiantes del conjunto de prueba.",
                            className="small text-muted mb-2",
                        ),
                        dcc.Graph(figure=grafico_shap_global(),
                                  config={"displayModeBar": False}),
                    ], className="py-2 px-3"),
                ], className="shadow-sm"),
            ], md=7),
        ]),
    ])


# --- Layout Q2 ---

def layout_q2():
    if not Q2_AVAILABLE:
        return dbc.Alert(
            ["Modelo Q2 no disponible. Ejecute ",
             html.Code("python data_science_2/export_model_q2.py"),
             " para generar los archivos del modelo."],
            color="warning", className="mt-3",
        )

    m = info_q2
    pct_bilingue = m["class_dist"]["bilingue"] / (
        m["class_dist"]["bilingue"] + m["class_dist"]["no_bilingue"]
    ) * 100

    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Clasificacion de colegios bilingues: "),
            "Con base en los puntajes Saber 11 y caracteristicas institucionales, "
            "el modelo clasifica si un colegio es ",
            html.Strong("Bilingue"),
            " en el Departamento del Cesar.",
            html.Br(),
            html.Small([
                f"Modelo: red neuronal [128,64,32] — {m['n_features']} variables — "
                f"AUC-ROC = {m['metrics']['auc_roc']:.3f} — "
                f"Recall = {m['metrics']['recall']:.3f} — "
                f"N entrenamiento = {m['n_train']:,} — "
                f"Bilingues en dataset = {pct_bilingue:.1f}%",
            ], className="text-muted"),
        ], color="light", dismissable=False, className="mb-3 py-2 banner-alert"),

        dbc.Row([
            dbc.Col([
                tarjeta_seccion("Puntajes Saber 11", "bi-journal-text", [
                    dbc.Row([
                        dbc.Col(campo("Ingles", dcc.Input(
                            id="q2-punt-ingles", type="number",
                            min=0, max=100, step=1, value=50,
                            className="form-control",
                        )), md=6),
                        dbc.Col(campo("Global", dcc.Input(
                            id="q2-punt-global", type="number",
                            min=0, max=500, step=1, value=250,
                            className="form-control",
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Lectura Critica", dcc.Input(
                            id="q2-punt-lectura", type="number",
                            min=0, max=100, step=1, value=50,
                            className="form-control",
                        )), md=6),
                        dbc.Col(campo("Matematicas", dcc.Input(
                            id="q2-punt-matematicas", type="number",
                            min=0, max=100, step=1, value=50,
                            className="form-control",
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Ciencias Naturales", dcc.Input(
                            id="q2-punt-ciencias", type="number",
                            min=0, max=100, step=1, value=50,
                            className="form-control",
                        )), md=6),
                        dbc.Col(campo("Sociales y Ciudadanas", dcc.Input(
                            id="q2-punt-sociales", type="number",
                            min=0, max=100, step=1, value=50,
                            className="form-control",
                        )), md=6),
                    ]),
                ]),

                tarjeta_seccion("Caracteristicas del Colegio", "bi-building", [
                    dbc.Row([
                        dbc.Col(campo("Naturaleza", dcc.Dropdown(
                            id="q2-naturaleza", options=NATURALEZA_OPTIONS,
                            value="OFICIAL", clearable=False,
                        )), md=6),
                        dbc.Col(campo("Jornada", dcc.Dropdown(
                            id="q2-jornada", options=JORNADA_OPTIONS,
                            value="UNICA", clearable=False,
                        )), md=6),
                    ]),
                ]),

                tarjeta_seccion("Contexto del Estudiante y Familia", "bi-people", [
                    dbc.Row([
                        dbc.Col(campo("Genero", dcc.Dropdown(
                            id="q2-genero", options=GENERO_OPTIONS,
                            value="F", clearable=False,
                        )), md=4),
                        dbc.Col(campo("Estrato", dcc.Dropdown(
                            id="q2-estrato", options=ESTRATO_OPTIONS_Q2,
                            value=2, clearable=False,
                        )), md=4),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Internet en el Hogar", dcc.Dropdown(
                            id="q2-internet", options=INTERNET_OPTIONS,
                            value="Si", clearable=False,
                        )), md=6),
                        dbc.Col(campo("Computador en el Hogar", dcc.Dropdown(
                            id="q2-computador", options=COMPUTADOR_OPTIONS,
                            value="Si", clearable=False,
                        )), md=6),
                    ]),
                ]),

                dbc.Button(
                    [html.I(className="bi bi-search me-2"), "Clasificar Colegio"],
                    id="q2-btn-predict", size="lg",
                    className="w-100 mb-2 fw-bold predict-btn",
                ),
                html.Small(
                    "La clasificacion se basa en patrones estadisticos historicos del Cesar.",
                    className="text-muted d-block text-center mb-4",
                ),
            ], md=5),

            dbc.Col([
                html.Div(id="q2-result-panel", children=[
                    html.Div([
                        html.Div(html.I(className="bi bi-building-check",
                                        style={"fontSize": "3.5rem", "color": CREMA2}),
                                 className="text-center mb-3"),
                        html.H5("Complete el formulario y presione Clasificar",
                                className="text-center text-muted"),
                        html.P("El modelo determinara si el colegio es Bilingue o No Bilingue.",
                               className="text-center text-muted small"),
                    ], className="py-5 px-3"),
                ]),

                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-info-circle me-2"),
                         "Como interpretar el resultado"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        html.P("El modelo devuelve una probabilidad de 0% a 100%:",
                               className="mb-2 small"),
                        html.Div([
                            html.Div([
                                html.Span(style={"display": "inline-block", "width": "12px",
                                                 "height": "12px", "borderRadius": "2px",
                                                 "backgroundColor": bg, "marginRight": "6px",
                                                 "verticalAlign": "middle"}),
                                html.Span(txt, className="small"),
                            ], className="mb-1")
                            for bg, txt in [
                                ("#fde8e8", "Probabilidad baja — No Bilingue"),
                                ("#e8f8f0", "Probabilidad alta — Bilingue"),
                            ]
                        ]),
                        html.Hr(className="my-2"),
                        html.P([html.Strong("Umbral de decision: "),
                                f"{loader_q2.threshold*100:.0f}% (optimizado por F1-score)."],
                               className="small mb-1"),
                        html.P([html.Strong("Nota: "),
                                f"Solo el {pct_bilingue:.1f}% de los colegios del Cesar "
                                "son bilingues. El modelo es conservador por eso."],
                               className="small mb-0"),
                    ], className="py-2 px-3"),
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-cpu me-2"), "Metricas del Modelo Q2"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.Div(f"{m['metrics']['auc_roc']:.3f}",
                                              className="metric-value", style={"color": VINOTINTO}),
                                     html.Div("AUC-ROC", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{m['metrics']['recall']:.3f}",
                                              className="metric-value", style={"color": "#1e8449"}),
                                     html.Div("Recall", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{m['metrics']['accuracy']:.3f}",
                                              className="metric-value", style={"color": "#d35400"}),
                                     html.Div("Accuracy", className="metric-label")],
                                    className="text-center"),
                            dbc.Col([html.Div(f"{m['n_features']}",
                                              className="metric-value", style={"color": "#1a5276"}),
                                     html.Div("Variables", className="metric-label")],
                                    className="text-center"),
                        ]),
                    ], className="py-2"),
                ], className="shadow-sm"),
            ], md=7),
        ]),
    ])


# --- Layout Q3 ---

def layout_q3():
    if not Q3_AVAILABLE:
        return dbc.Alert(
            ["Modelo Q3 no disponible. Ejecute ",
             html.Code("python export_model_q3.py"),
             " para generar los archivos del modelo."],
            color="warning", className="mt-3",
        )

    m = info_q3
    ps = m["metrics"]["per_subject"]

    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Prediccion de puntajes por area: "),
            "Con base en el contexto escolar y familiar del estudiante, "
            "el modelo estima el puntaje esperado en cada una de las ",
            html.Strong("5 areas de la Prueba Saber 11"),
            " para el Departamento del Cesar.",
            html.Br(),
            html.Small([
                f"Modelo: arch_4 (Dense+BN) — {m['n_features']} variables — "
                f"MAE promedio = {m['metrics']['mae_mean']:.2f} pts — "
                f"R² promedio = {m['metrics']['r2_mean']:.3f} — "
                f"N entrenamiento = {m['n_train']:,}",
                html.Br(),
                html.Strong("Nota: "),
                f"Con R²={m['metrics']['r2_mean']:.2f}, el modelo explica el {m['metrics']['r2_mean']*100:.0f}% "
                "de la varianza por area. Las predicciones tienden al promedio historico del Cesar; "
                "la diferenciacion entre perfiles es limitada. Para estimacion del puntaje global se "
                "recomienda usar la Pregunta 1 (R²=0.34).",
            ], className="text-muted"),
        ], color="light", dismissable=False, className="mb-3 py-2 banner-alert"),

        dbc.Row([

            # ── LEFT: Input form ─────────────────────────────────────────
            dbc.Col([
                tarjeta_seccion("Periodo del Examen", "bi-calendar3", [
                    campo("Periodo", dcc.Dropdown(
                        id="q3-periodo", options=opciones_periodo,
                        value=20221, clearable=False,
                    )),
                ]),

                tarjeta_seccion("Datos del Estudiante", "bi-person", [
                    dbc.Row([
                        dbc.Col(campo("Genero", dcc.Dropdown(
                            id="q3-genero",
                            options=[{"label": "Masculino", "value": True},
                                     {"label": "Femenino",  "value": False}],
                            value=True, clearable=False,
                        )), md=6),
                        dbc.Col(campo("Nacionalidad", dcc.Dropdown(
                            id="q3-nacionalidad",
                            options=[
                                {"label": "Colombia",  "value": "COLOMBIA"},
                                {"label": "Venezuela", "value": "VENEZUELA"},
                                {"label": "Ecuador",   "value": "ECUADOR"},
                                {"label": "España",    "value": "ESPAÑA"},
                            ],
                            value="COLOMBIA", clearable=False,
                        )), md=6),
                    ]),
                    campo("Municipio de Residencia", dcc.Dropdown(
                        id="q3-mcpio-residencia", options=opciones_mcpio,
                        value=20001, clearable=False,
                    )),
                ]),

                tarjeta_seccion("Contexto Familiar", "bi-house", [
                    dbc.Row([
                        dbc.Col(campo("Estrato de Vivienda", dcc.Dropdown(
                            id="q3-estrato",
                            options=[
                                {"label": "Sin Estrato", "value": "Sin Estrato"},
                                {"label": "Estrato 1",   "value": "Estrato 1"},
                                {"label": "Estrato 2",   "value": "Estrato 2"},
                                {"label": "Estrato 3",   "value": "Estrato 3"},
                                {"label": "Estrato 4",   "value": "Estrato 4"},
                                {"label": "Estrato 5",   "value": "Estrato 5"},
                                {"label": "Estrato 6",   "value": "Estrato 6"},
                            ],
                            value="Estrato 1", clearable=False,
                        )), md=6),
                        dbc.Col(campo("Personas en el Hogar", dcc.Dropdown(
                            id="q3-personas", options=PERSONAS_OPTIONS_Q3,
                            value="3 a 4", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Cuartos en el Hogar", dcc.Dropdown(
                            id="q3-cuartos", options=CUARTOS_OPTIONS_Q3,
                            value="3", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Educacion de la Madre",
                            dcc.Dropdown(id="q3-edu-madre", options=opciones_educacion,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                        dbc.Col(campo("Educacion del Padre",
                            dcc.Dropdown(id="q3-edu-padre", options=opciones_educacion,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                    ]),
                    html.Label("Recursos del hogar",
                               className="form-label fw-semibold mb-2 input-label"),
                    dbc.Checklist(
                        id="q3-recursos",
                        options=[
                            {"label": "Computador", "value": "computador"},
                            {"label": "Internet",   "value": "internet"},
                            {"label": "Automovil",  "value": "automovil"},
                            {"label": "Lavadora",   "value": "lavadora"},
                        ],
                        value=["computador", "internet"],
                        switch=True, inline=False,
                    ),
                ]),

                tarjeta_seccion("Contexto Escolar", "bi-building", [
                    campo("Municipio del Colegio", dcc.Dropdown(
                        id="q3-mcpio-cole", options=opciones_mcpio,
                        value=20001, clearable=False,
                    )),
                    campo("Colegio", dcc.Dropdown(
                        id="q3-colegio", options=[],
                        placeholder="Seleccione el municipio primero...", clearable=True,
                    )),
                    dbc.Alert(
                        [html.I(className="bi bi-magic me-2"),
                         "Al seleccionar el colegio los campos se completan automaticamente."],
                        color="light", className="py-1 px-2 mb-3 small banner-alert",
                        dismissable=False,
                    ),
                    dbc.Row([
                        dbc.Col(campo("Area", dcc.Dropdown(
                            id="q3-area",
                            options=[{"label": "Urbano", "value": True},
                                     {"label": "Rural",  "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(campo("Calendario", dcc.Dropdown(
                            id="q3-calendario",
                            options=[{"label": "Calendario A", "value": True},
                                     {"label": "Calendario B", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(campo("Naturaleza", dcc.Dropdown(
                            id="q3-naturaleza",
                            options=[{"label": "Oficial (Publica)",   "value": True},
                                     {"label": "No Oficial (Privada)", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                    ]),
                    dbc.Row([
                        dbc.Col(campo("Jornada", dcc.Dropdown(
                            id="q3-jornada",
                            options=[
                                {"label": "Unica",    "value": "UNICA"},
                                {"label": "Manana",   "value": "MAÑANA"},
                                {"label": "Tarde",    "value": "TARDE"},
                                {"label": "Completa", "value": "COMPLETA"},
                                {"label": "Sabatina", "value": "SABATINA"},
                                {"label": "Noche",    "value": "NOCHE"},
                            ],
                            value="UNICA", clearable=False,
                        )), md=4),
                        dbc.Col(campo("Caracter", dcc.Dropdown(
                            id="q3-caracter",
                            options=[
                                {"label": "Academico",           "value": "ACADÉMICO"},
                                {"label": "Tecnico",             "value": "TÉCNICO"},
                                {"label": "Tecnico / Academico", "value": "TÉCNICO/ACADÉMICO"},
                            ],
                            value="ACADÉMICO", clearable=False,
                        )), md=4),
                        dbc.Col(campo("Bilingue", dcc.Dropdown(
                            id="q3-bilingue",
                            options=[{"label": "No", "value": False},
                                     {"label": "Si", "value": True}],
                            value=False, clearable=False,
                        )), md=4),
                    ]),
                ]),

                dbc.Button(
                    [html.I(className="bi bi-grid-3x2 me-2"), "Predecir por Area"],
                    id="q3-btn-predict", size="lg",
                    className="w-100 mb-2 fw-bold predict-btn",
                ),
                html.Small(
                    "La prediccion es una estimacion basada en datos historicos del Cesar.",
                    className="text-muted d-block text-center mb-4",
                ),
            ], md=5),

            # ── RIGHT: Results ───────────────────────────────────────────
            dbc.Col([
                html.Div(id="q3-result-panel", children=[
                    html.Div([
                        html.Div(html.I(className="bi bi-grid-3x2-gap",
                                        style={"fontSize": "3.5rem", "color": CREMA2}),
                                 className="text-center mb-3"),
                        html.H5("Complete el formulario y presione Predecir",
                                className="text-center text-muted"),
                        html.P("El modelo estimara los puntajes en las 5 areas de la Saber 11.",
                               className="text-center text-muted small"),
                    ], className="py-5 px-3"),
                ]),

                # Reference card (static)
                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-info-circle me-2"),
                         "Promedios historicos por area (Cesar)"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span(style={
                                        "display": "inline-block", "width": "10px",
                                        "height": "10px", "borderRadius": "2px",
                                        "backgroundColor": Q3_TARGET_COLORS[col],
                                        "marginRight": "5px", "verticalAlign": "middle",
                                    }),
                                    html.Span(label, className="small fw-semibold"),
                                    html.Span(f"  {ps[col]['mean']:.1f} pts",
                                              className="small text-muted ms-1"),
                                ], className="mb-1")
                                for col, label in zip(Q3_TARGET_COLS, Q3_TARGET_LABELS)
                            ]),
                        ]),
                        html.Hr(className="my-2"),
                        html.P([html.Strong("Rango de confianza: "),
                                f"MAE promedio ±{m['metrics']['mae_mean']:.1f} pts."],
                               className="small mb-0"),
                    ], className="py-2 px-3"),
                ], className="mb-3 shadow-sm"),

                # Model metrics (static)
                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-cpu me-2"), "Metricas del Modelo Q3"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(f"{m['metrics']['mae_mean']:.2f}",
                                         className="metric-value", style={"color": "#c0392b"}),
                                html.Div("MAE medio (pts)", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div(f"{m['metrics']['r2_mean']:.3f}",
                                         className="metric-value", style={"color": "#1e8449"}),
                                html.Div("R² medio", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div(f"{m['n_features']}",
                                         className="metric-value", style={"color": VINOTINTO}),
                                html.Div("Variables", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div("5",
                                         className="metric-value", style={"color": "#8e44ad"}),
                                html.Div("Salidas", className="metric-label"),
                            ], className="text-center"),
                        ]),
                    ], className="py-2"),
                ], className="shadow-sm"),
            ], md=7),
        ]),
    ])


# --- Layout principal ---

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.Div([
            html.Div(
                html.Img(src="/assets/images/min_educacion_logo.png",
                         style={"height": "44px", "display": "block"}),
                style={"backgroundColor": CREMA, "borderRadius": "8px",
                       "padding": "5px 10px", "marginRight": "18px",
                       "display": "inline-block", "verticalAlign": "middle"},
            ),
            html.Div([
                html.H4("Predictor de Puntaje Saber 11",
                        className="mb-0 fw-bold text-white"),
                html.P("Departamento del Cesar — Modelo de Red Neuronal",
                       className="mb-0 small",
                       style={"color": "rgba(255,255,255,0.65)"}),
            ], className="d-inline-block align-middle"),
        ], className="py-3 d-flex align-items-center")),
    ], className="header-row mb-4"),

    dbc.Row([
        dbc.Col(dbc.Tabs([
            dbc.Tab(label="Pregunta 1: Puntaje Global",
                    tab_id="q1", label_style={"fontWeight": "600"}),
            dbc.Tab(
                label="Pregunta 2: Clasificacion Bilingue",
                tab_id="q2",
                label_style={"fontWeight": "600"},
                disabled=not Q2_AVAILABLE,
            ),
            dbc.Tab(
                label="Pregunta 3: Puntajes por Area",
                tab_id="q3",
                label_style={"fontWeight": "600"},
                disabled=not Q3_AVAILABLE,
            ),
        ], id="tabs", active_tab="q1")),
    ], className="mb-3"),

    html.Div(id="tab-content"),

    html.Hr(className="mt-4"),
    html.Div(
        html.Small([
            "Desarrollado para el ",
            html.Strong("Ministerio de Educacion Nacional"),
            " — Datos ICFES Saber 11, Departamento del Cesar — ",
            html.Span("Universidad de los Andes — IIND 4130", className="text-muted"),
        ]),
        className="text-center py-2 text-muted",
    ),

], fluid=True, className="px-4")


# --- Callbacks ---

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def mostrar_tab(tab):
    if tab == "q2":
        return layout_q2()
    if tab == "q3":
        return layout_q3()
    return layout_q1()


@app.callback(
    Output("result-panel", "children"),
    Input("btn-predict", "n_clicks"),
    State("in-periodo",           "value"),
    State("in-genero",            "value"),
    State("in-nacionalidad",      "value"),
    State("in-mcpio-residencia",  "value"),
    State("in-estrato",           "value"),
    State("in-personas",          "value"),
    State("in-cuartos",           "value"),
    State("in-edu-madre",         "value"),
    State("in-edu-padre",         "value"),
    State("in-recursos",          "value"),
    State("in-mcpio-cole",        "value"),
    State("in-area",              "value"),
    State("in-calendario",        "value"),
    State("in-naturaleza",        "value"),
    State("in-jornada",           "value"),
    State("in-caracter",          "value"),
    State("in-bilingue",          "value"),
    State("in-colegio",           "value"),
    prevent_initial_call=True,
)
def predecir_q1(
    n_clicks,
    periodo, genero, nacionalidad, mcpio_res,
    estrato, personas, cuartos, edu_madre, edu_padre, recursos,
    mcpio_cole, area, calendario, naturaleza, jornada, caracter, bilingue,
    colegio,
):
    if not n_clicks:
        return no_update

    recursos = recursos or []
    entradas = {
        "periodo":                       int(periodo),
        "cole_area_urbano":              bool(area),
        "cole_bilingue":                 bool(bilingue),
        "cole_calendario_a":             bool(calendario),
        "cole_oficial":                  bool(naturaleza),
        "estu_masculino":                bool(genero),
        "fami_tieneautomovil":           "automovil"  in recursos,
        "fami_tienecomputador":          "computador" in recursos,
        "fami_tieneinternet":            "internet"   in recursos,
        "fami_tienelavadora":            "lavadora"   in recursos,
        "cole_caracter":                 caracter,
        "cole_jornada":                  jornada,
        "estu_nacionalidad":             nacionalidad,
        "fami_cuartoshogar":             cuartos,
        "fami_educacionmadre":           edu_madre,
        "fami_educacionpadre":           edu_padre,
        "fami_estratovivienda":          estrato,
        "fami_personashogar":            personas,
        "cole_cod_mcpio_ubicacion":      int(mcpio_cole),
        "estu_cod_reside_mcpio":         int(mcpio_res),
        "cole_cod_dane_establecimiento": int(colegio) if colegio else None,
        "cole_cod_dane_sede":            None,
    }

    try:
        score = float(np.clip(loader.predict(entradas), 0, 500))
    except Exception as e:
        return dbc.Alert(f"Error en la prediccion: {e}", color="danger")

    label, color = nivel_puntaje(score)
    mae        = info["metrics"]["mae"]
    mean_score = info["target_mean"]
    delta      = score - mean_score
    texto_delta = (
        f"+{delta:.0f} pts por encima del promedio historico ({mean_score:.0f} pts)"
        if delta >= 0 else
        f"{delta:.0f} pts por debajo del promedio historico ({mean_score:.0f} pts)"
    )
    color_delta = "#1e8449" if delta >= 0 else "#c0392b"

    try:
        factores = loader.explain_groups(entradas, top_n=5)
    except Exception:
        factores = []

    def item_factor(texto_factor, impacto):
        if impacto >= 0:
            icono, ci, signo = "bi-arrow-up-circle-fill", "#1e8449", f"+{impacto:.1f}"
        else:
            icono, ci, signo = "bi-arrow-down-circle-fill", "#c0392b", f"{impacto:.1f}"
        return html.Li([
            html.I(className=f"bi {icono} me-2", style={"color": ci}),
            html.Strong(f"{signo} pts"),
            html.Span(f" — {texto_factor}", className="text-muted"),
        ], className="mb-1 small")

    contenido_factores = (
        html.Ul([item_factor(g["label"], g["impact"]) for g in factores],
                className="mb-0 ps-2")
        if factores else
        html.P("No se identificaron factores destacables.", className="text-muted small mb-0")
    )

    return [
        dbc.Card([
            dbc.CardBody(dcc.Graph(figure=grafico_gauge_q1(score),
                                   config={"displayModeBar": False}),
                         className="py-2"),
        ], className="mb-3 shadow"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-pin-map me-2"), "Posicion en la escala Saber 11"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                dcc.Graph(figure=grafico_barra_contexto(score),
                          config={"displayModeBar": False}),
                html.P(texto_delta, className="text-center small mt-1",
                       style={"color": color_delta}),
            ], className="py-2 px-2"),
        ], className="mb-3 shadow-sm"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-clipboard-check me-2"), "Resumen del Resultado"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{score:.0f}", className="big-score", style={"color": color}),
                        html.Div("puntos", className="text-muted small"),
                    ], className="text-center", md=3),
                    dbc.Col([
                        html.P([html.Strong("Nivel: "), label], className="mb-1"),
                        html.P([html.Strong("Rango probable: "),
                                f"{max(0, score-mae):.0f} — {min(500, score+mae):.0f} pts"],
                               className="mb-1 small text-muted"),
                        html.P([html.Strong("Comparado con el Cesar: "),
                                texto_delta.split(":")[0] if ":" in texto_delta else texto_delta],
                               className="mb-0 small text-muted"),
                    ], md=9),
                ]),
            ], className="py-2 px-3"),
        ], className="mb-3 shadow-sm"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-lightbulb me-2"), "Factores Clave Detectados"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                html.P("Impacto estimado de cada factor vs. el estudiante promedio del Cesar:",
                       className="small text-muted mb-2"),
                contenido_factores,
            ], className="py-2 px-3"),
        ], className="shadow-sm"),
    ]


@app.callback(
    Output("in-colegio", "options"),
    Output("in-colegio", "value"),
    Input("in-mcpio-cole", "value"),
)
def actualizar_colegios(municipio):
    if not municipio:
        return [], None
    colegios = COLEGIOS_BY_MCPIO.get(int(municipio), [])
    opciones = sorted(
        [{"label": s["nombre"].title(), "value": s["code"]} for s in colegios],
        key=lambda x: x["label"],
    )
    return opciones, None


@app.callback(
    Output("in-area",       "value"),
    Output("in-bilingue",   "value"),
    Output("in-calendario", "value"),
    Output("in-naturaleza", "value"),
    Output("in-caracter",   "value"),
    Output("in-jornada",    "value"),
    Input("in-colegio", "value"),
    prevent_initial_call=True,
)
def autocompletar_colegio(codigo):
    if not codigo:
        return no_update, no_update, no_update, no_update, no_update, no_update
    datos = COLEGIOS.get(str(codigo))
    if not datos:
        return no_update, no_update, no_update, no_update, no_update, no_update
    return (
        datos["area_urbano"],
        datos["bilingue"],
        datos["calendario_a"],
        datos["oficial"],
        datos["caracter"],
        datos["jornada"],
    )


@app.callback(
    Output("q2-result-panel", "children"),
    Input("q2-btn-predict", "n_clicks"),
    State("q2-punt-ingles",      "value"),
    State("q2-punt-global",      "value"),
    State("q2-punt-lectura",     "value"),
    State("q2-punt-matematicas", "value"),
    State("q2-punt-ciencias",    "value"),
    State("q2-punt-sociales",    "value"),
    State("q2-estrato",          "value"),
    State("q2-naturaleza",       "value"),
    State("q2-jornada",          "value"),
    State("q2-genero",           "value"),
    State("q2-internet",         "value"),
    State("q2-computador",       "value"),
    prevent_initial_call=True,
)
def clasificar_q2(
    n_clicks,
    punt_ingles, punt_global, punt_lectura, punt_matematicas,
    punt_ciencias, punt_sociales,
    estrato, naturaleza, jornada, genero, internet, computador,
):
    if not n_clicks:
        return no_update

    if not Q2_AVAILABLE:
        return dbc.Alert("Modelo Q2 no disponible.", color="warning")

    entradas = {
        "punt_ingles":              float(punt_ingles or 0),
        "punt_global":              float(punt_global or 0),
        "punt_lectura_critica":     float(punt_lectura or 0),
        "punt_matematicas":         float(punt_matematicas or 0),
        "punt_c_naturales":         float(punt_ciencias or 0),
        "punt_sociales_ciudadanas": float(punt_sociales or 0),
        "fami_estratovivienda":     float(estrato or 2),
        "cole_naturaleza":          naturaleza or "OFICIAL",
        "cole_jornada":             jornada or "UNICA",
        "estu_genero":              genero or "F",
        "fami_tieneinternet":       internet or "No",
        "fami_tienecomputador":     computador or "No",
    }

    try:
        resultado = loader_q2.predict(entradas)
    except Exception as e:
        return dbc.Alert(f"Error en la clasificacion: {e}", color="danger")

    prob         = resultado["probability"]
    es_bilingue  = resultado["is_bilingual"]
    label        = resultado["label"]
    color        = "#1e8449" if es_bilingue else "#c0392b"

    try:
        factores = loader_q2.explain_groups(entradas, top_n=5)
    except Exception:
        factores = []

    def item_factor_q2(texto_factor, impacto):
        if impacto >= 0:
            icono, ci, signo = "bi-arrow-up-circle-fill", "#1e8449", f"+{impacto:.3f}"
        else:
            icono, ci, signo = "bi-arrow-down-circle-fill", "#c0392b", f"{impacto:.3f}"
        return html.Li([
            html.I(className=f"bi {icono} me-2", style={"color": ci}),
            html.Strong(signo),
            html.Span(f" prob — {texto_factor}", className="text-muted"),
        ], className="mb-1 small")

    contenido_factores = (
        html.Ul([item_factor_q2(g["label"], g["impact"]) for g in factores],
                className="mb-0 ps-2")
        if factores else
        html.P("No se identificaron factores destacables.", className="text-muted small mb-0")
    )

    return [
        dbc.Card([
            dbc.CardBody([
                html.Div(badge_bilingue(es_bilingue), className="text-center mb-3"),
                dcc.Graph(figure=grafico_gauge_q2(prob, es_bilingue),
                          config={"displayModeBar": False}),
            ], className="py-2"),
        ], className="mb-3 shadow"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-clipboard-check me-2"), "Resumen de Clasificacion"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{prob*100:.1f}%",
                                 className="big-score", style={"color": color}),
                        html.Div("probabilidad", className="text-muted small"),
                    ], className="text-center", md=4),
                    dbc.Col([
                        html.P([html.Strong("Clasificacion: "), label], className="mb-1"),
                        html.P([html.Strong("Umbral: "),
                                f"{loader_q2.threshold*100:.0f}% (F1-optimo)"],
                               className="mb-1 small text-muted"),
                        html.P([html.Strong("AUC-ROC del modelo: "),
                                f"{info_q2['metrics']['auc_roc']:.3f}"],
                               className="mb-0 small text-muted"),
                    ], md=8),
                ]),
            ], className="py-2 px-3"),
        ], className="mb-3 shadow-sm"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-lightbulb me-2"), "Factores Clave Detectados"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                html.P("Cambio en probabilidad al reemplazar cada factor con el promedio del dataset:",
                       className="small text-muted mb-2"),
                contenido_factores,
            ], className="py-2 px-3"),
        ], className="shadow-sm"),
    ]


@app.callback(
    Output("q3-result-panel", "children"),
    Input("q3-btn-predict", "n_clicks"),
    State("q3-periodo",          "value"),
    State("q3-genero",           "value"),
    State("q3-nacionalidad",     "value"),
    State("q3-mcpio-residencia", "value"),
    State("q3-estrato",          "value"),
    State("q3-personas",         "value"),
    State("q3-cuartos",          "value"),
    State("q3-edu-madre",        "value"),
    State("q3-edu-padre",        "value"),
    State("q3-recursos",         "value"),
    State("q3-mcpio-cole",       "value"),
    State("q3-area",             "value"),
    State("q3-calendario",       "value"),
    State("q3-naturaleza",       "value"),
    State("q3-jornada",          "value"),
    State("q3-caracter",         "value"),
    State("q3-bilingue",         "value"),
    State("q3-colegio",          "value"),
    prevent_initial_call=True,
)
def predecir_q3(
    n_clicks,
    periodo, genero, nacionalidad, mcpio_res,
    estrato, personas, cuartos, edu_madre, edu_padre, recursos,
    mcpio_cole, area, calendario, naturaleza, jornada, caracter, bilingue,
    colegio,
):
    if not n_clicks:
        return no_update
    if not Q3_AVAILABLE:
        return dbc.Alert("Modelo Q3 no disponible.", color="warning")

    recursos = recursos or []
    entradas = {
        "periodo":                       int(periodo),
        "cole_area_urbano":              bool(area),
        "cole_bilingue":                 bool(bilingue),
        "cole_calendario_a":             bool(calendario),
        "cole_oficial":                  bool(naturaleza),
        "estu_masculino":                bool(genero),
        "fami_tieneautomovil":           "automovil"  in recursos,
        "fami_tienecomputador":          "computador" in recursos,
        "fami_tieneinternet":            "internet"   in recursos,
        "fami_tienelavadora":            "lavadora"   in recursos,
        "cole_caracter":                 caracter,
        "cole_jornada":                  jornada,
        "estu_nacionalidad":             nacionalidad,
        "fami_cuartoshogar":             cuartos,
        "fami_educacionmadre":           edu_madre,
        "fami_educacionpadre":           edu_padre,
        "fami_estratovivienda":          estrato,
        "fami_personashogar":            personas,
        "cole_cod_mcpio_ubicacion":      int(mcpio_cole),
        "estu_cod_reside_mcpio":         int(mcpio_res),
        "cole_cod_dane_establecimiento": int(colegio) if colegio else None,
        "cole_cod_dane_sede":            None,
    }

    try:
        scores = loader_q3.predict(entradas)
    except Exception as e:
        return dbc.Alert(f"Error en la prediccion: {e}", color="danger")

    ps = info_q3["metrics"]["per_subject"]

    return [
        # Radar chart
        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-hexagon me-2"), "Perfil de Puntajes por Area"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody(
                dcc.Graph(figure=grafico_radar_q3(scores),
                          config={"displayModeBar": False}),
                className="py-2",
            ),
        ], className="mb-3 shadow"),

        # Per-subject score cards
        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-clipboard-check me-2"), "Puntajes Estimados"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                dbc.Row(tarjetas_materias(scores), className="g-2"),
                html.P(
                    f"El rango de confianza de cada puntaje es ± el MAE por materia.",
                    className="text-muted small text-center mt-2 mb-0",
                ),
            ], className="py-2 px-2"),
        ], className="mb-3 shadow-sm"),

        # Detailed comparison table
        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-table me-2"), "Comparacion con el Promedio del Cesar"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                html.Div([
                    dbc.Row([
                        dbc.Col(html.Strong("Area", className="small"), width=4),
                        dbc.Col(html.Strong("Prediccion", className="small"), width=2,
                                className="text-center"),
                        dbc.Col(html.Strong("Promedio", className="small"), width=2,
                                className="text-center"),
                        dbc.Col(html.Strong("Diferencia", className="small"), width=2,
                                className="text-center"),
                        dbc.Col(html.Strong("MAE", className="small"), width=2,
                                className="text-center"),
                    ], className="mb-1 pb-1 border-bottom"),
                ] + [
                    dbc.Row([
                        dbc.Col(html.Span([
                            html.Span(style={"display": "inline-block", "width": "8px",
                                             "height": "8px", "borderRadius": "50%",
                                             "backgroundColor": Q3_TARGET_COLORS[col],
                                             "marginRight": "5px"}),
                            label,
                        ], className="small"), width=4),
                        dbc.Col(html.Span(f"{scores[col]:.1f}", className="small fw-bold",
                                          style={"color": Q3_TARGET_COLORS[col]}),
                                width=2, className="text-center"),
                        dbc.Col(html.Span(f"{ps[col]['mean']:.1f}", className="small text-muted"),
                                width=2, className="text-center"),
                        dbc.Col(
                            html.Span(
                                f"{scores[col] - ps[col]['mean']:+.1f}",
                                className="small fw-semibold",
                                style={"color": "#1e8449" if scores[col] >= ps[col]["mean"] else "#c0392b"},
                            ),
                            width=2, className="text-center",
                        ),
                        dbc.Col(html.Span(f"±{ps[col]['mae']:.1f}", className="small text-muted"),
                                width=2, className="text-center"),
                    ], className="mb-1 py-1 border-bottom border-light")
                    for col, label in zip(Q3_TARGET_COLS, Q3_TARGET_LABELS)
                ]),
            ], className="py-2 px-3"),
        ], className="mb-3 shadow-sm"),

    ]


@app.callback(
    Output("q3-colegio", "options"),
    Output("q3-colegio", "value"),
    Input("q3-mcpio-cole", "value"),
)
def actualizar_colegios_q3(municipio):
    if not municipio:
        return [], None
    colegios = COLEGIOS_BY_MCPIO.get(int(municipio), [])
    opciones = sorted(
        [{"label": s["nombre"].title(), "value": s["code"]} for s in colegios],
        key=lambda x: x["label"],
    )
    return opciones, None


@app.callback(
    Output("q3-area",       "value"),
    Output("q3-bilingue",   "value"),
    Output("q3-calendario", "value"),
    Output("q3-naturaleza", "value"),
    Output("q3-caracter",   "value"),
    Output("q3-jornada",    "value"),
    Input("q3-colegio", "value"),
    prevent_initial_call=True,
)
def autocompletar_colegio_q3(codigo):
    if not codigo:
        return no_update, no_update, no_update, no_update, no_update, no_update
    datos = COLEGIOS.get(str(codigo))
    if not datos:
        return no_update, no_update, no_update, no_update, no_update, no_update
    return (
        datos["area_urbano"], datos["bilingue"], datos["calendario_a"],
        datos["oficial"], datos["caracter"], datos["jornada"],
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
