"""
app.py
Dashboard del Ministerio de Educación - Predictor de Puntaje ICFES
Prueba Saber 11 - Departamento del Cesar

Preguntas de negocio:
  Q1: ¿Cuál es el puntaje global esperado dado el contexto del estudiante y su colegio?
  Q2: (Pendiente)
  Q3: (Pendiente)

Deploy: TODO: cmd for dash
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

# ── Brand colors ───────────────────────────────────────────────────────────────
VINOTINTO  = "#7B1C2B"
VINOTINTO2 = "#5E1520"   # hover / dark variant
CREMA      = "#FAF5EC"
CREMA2     = "#F0E8D5"   # borders / accents

# ── App initialization ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
loader = ModelLoader(MODEL_DIR)
info = loader.model_info

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    title="Predictor ICFES - Ministerio de Educación",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # Expose Flask server for gunicorn

# ── Helpers ────────────────────────────────────────────────────────────────────

def score_level(score: float):
    if score < 200:
        return "Bajo", "#c0392b"
    elif score < 250:
        return "Medio-Bajo", "#d35400"
    elif score < 300:
        return "Medio", "#b7950b"
    elif score < 350:
        return "Medio-Alto", "#1e8449"
    else:
        return "Alto", "#1a5276"


def make_gauge(score: float) -> go.Figure:
    label, color = score_level(score)
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
            "axis": {
                "range": [0, 500], "tickwidth": 1, "tickcolor": "#888",
                "tickvals": [0, 100, 200, 250, 300, 350, 400, 500],
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": CREMA2,
            "steps": [
                {"range": [0,   200], "color": "#fde8e8"},
                {"range": [200, 250], "color": "#fef3e2"},
                {"range": [250, 300], "color": "#fefbe6"},
                {"range": [300, 350], "color": "#e8f8f0"},
                {"range": [350, 500], "color": "#d6eaf8"},
            ],
            "threshold": {
                "line": {"color": "#555", "width": 3},
                "thickness": 0.8,
                "value": info["target_mean"],
            },
        },
        title={
            "text": f"Puntaje Predicho<br><span style='font-size:0.9em;color:{color}'>{label}</span>",
            "font": {"size": 18},
        },
    ))
    fig.add_annotation(
        text=f"Rango probable: {max(0, score - mae):.0f} – {min(500, score + mae):.0f} pts",
        x=0.5, y=-0.08, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color="#888"),
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=55),
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_global_shap_chart() -> go.Figure:
    groups = loader.shap_global
    if not groups:
        return go.Figure()
    labels = [g["label"] for g in reversed(groups)]
    values = [g["importance"] for g in reversed(groups)]
    max_v  = max(values) if values else 1
    colors = [
        f"rgba(123,28,43,{0.35 + 0.65 * (v / max_v):.2f})"
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} pts" for v in values],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="%{y}: %{x:.2f} pts<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Media |SHAP| (pts)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=30),
        height=max(280, len(groups) * 28 + 60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_context_bar(score: float) -> go.Figure:
    mean = info["target_mean"]
    mae  = info["metrics"]["mae"]
    fig  = go.Figure()

    bands = [
        (0,   200, "#fde8e8", "Bajo"),
        (200, 250, "#fef3e2", "Medio-Bajo"),
        (250, 300, "#fefbe6", "Medio"),
        (300, 350, "#e8f8f0", "Medio-Alto"),
        (350, 500, "#d6eaf8", "Alto"),
    ]
    for lo, hi, clr, lbl in bands:
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1,
                      fillcolor=clr, line_width=0, layer="below")
        fig.add_annotation(x=(lo + hi) / 2, y=0.5, text=lbl, showarrow=False,
                           font=dict(size=10, color="#888"), yref="paper")

    fig.add_vline(x=mean, line_dash="dash", line_color="#aaa", line_width=1.5,
                  annotation_text=f"Promedio hist. ({mean:.0f})",
                  annotation_position="top left",
                  annotation_font_size=11,
                  annotation_yshift=12)
    fig.add_shape(
        type="rect",
        x0=max(0, score - mae), x1=min(500, score + mae),
        y0=0.2, y1=0.8,
        fillcolor=f"rgba(123,28,43,0.18)",
        line=dict(color=VINOTINTO, width=1.5), layer="above",
    )
    fig.add_scatter(
        x=[score], y=[0.5], mode="markers",
        marker=dict(size=18, color=VINOTINTO, symbol="diamond"),
        showlegend=False,
    )
    fig.update_layout(
        xaxis=dict(range=[0, 500], title="Puntaje Global",
                   tickvals=[0, 100, 200, 250, 300, 350, 400, 500]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=110,
        margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ── Dropdown options ───────────────────────────────────────────────────────────
mcpio_options = [
    {"label": f"{name} ({code})", "value": code}
    for code, name in sorted(MUNICIPIOS_CESAR.items(), key=lambda x: x[1])
]
periodo_options = [
    {"label": label, "value": code}
    for code, label in sorted(PERIODOS_LABELS.items())
]
educacion_options = [
    {"label": label, "value": raw}
    for raw, label in EDUCATION_LABELS.items()
]

# ── Layout helpers ─────────────────────────────────────────────────────────────

def labeled_input(label: str, component):
    return html.Div([
        html.Label(label, className="form-label fw-semibold mb-1 input-label"),
        component,
    ], className="mb-3")


def section_card(title: str, icon_class: str, children):
    return dbc.Card([
        dbc.CardHeader(
            html.H6(
                [html.I(className=f"bi {icon_class} me-2"), title],
                className="mb-0 fw-bold section-title",
            ),
            className="section-header",
        ),
        dbc.CardBody(children, className="py-3 px-3"),
    ], className="mb-3 shadow-sm")


def icon_label(icon_class: str, text: str, color: str = VINOTINTO):
    return html.Span([
        html.I(className=f"bi {icon_class} me-1", style={"color": color}),
        text,
    ])


# ── App Layout ─────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            html.Div(
                html.Img(
                    src="/assets/images/min_educacion_logo.png",
                    style={"height": "44px", "display": "block"},
                ),
                style={
                    "backgroundColor": CREMA,
                    "borderRadius": "8px",
                    "padding": "5px 10px",
                    "marginRight": "18px",
                    "display": "inline-block",
                    "verticalAlign": "middle",
                },
            ),
            html.Div([
                html.H4("Predictor de Puntaje Saber 11",
                        className="mb-0 fw-bold text-white"),
                html.P("Departamento del Cesar · Modelo de Red Neuronal",
                       className="mb-0 small", style={"color": "rgba(255,255,255,0.65)"}),
            ], className="d-inline-block align-middle"),
        ], className="py-3 d-flex align-items-center")),
    ], className="header-row mb-4"),

    # ── Navigation tabs ──────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Tabs([
            dbc.Tab(label="Pregunta 1: Puntaje Global", tab_id="q1",
                    label_style={"fontWeight": "600"}),
            dbc.Tab(label="Pregunta 2: Próximamente", tab_id="q2", disabled=True),
            dbc.Tab(label="Pregunta 3: Próximamente", tab_id="q3", disabled=True),
        ], id="tabs", active_tab="q1")),
    ], className="mb-3"),

    # ── Q1 Tab content ───────────────────────────────────────────────────────
    html.Div(id="tab-content", children=[

        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("¿Qué predice este modelo? "),
            "Con base en las características del estudiante, su familia y su colegio, la red neuronal estima el ",
            html.Strong("Puntaje Global"),
            " esperado en la Prueba Saber 11 para el Departamento del Cesar. ",
            html.Br(),
            html.Small([
                f"Modelo: arch_D_deep · {info['n_features']} variables · ",
                f"MAE = {info['metrics']['mae']:.1f} pts · ",
                f"R² = {info['metrics']['r2']:.3f} · ",
                f"N entrenamiento = {info['n_train']:,}",
            ], className="text-muted"),
        ], color="light", dismissable=False, className="mb-3 py-2 banner-alert"),

        dbc.Row([

            # ── LEFT: Input form ─────────────────────────────────────────────
            dbc.Col([

                section_card("Período del Examen", "bi-calendar3", [
                    labeled_input("Período", dcc.Dropdown(
                        id="in-periodo", options=periodo_options,
                        value=20221, clearable=False,
                    )),
                ]),

                section_card("Datos del Estudiante", "bi-person", [
                    dbc.Row([
                        dbc.Col(labeled_input("Género", dcc.Dropdown(
                            id="in-genero",
                            options=[{"label": "Masculino", "value": True},
                                     {"label": "Femenino",  "value": False}],
                            value=True, clearable=False,
                        )), md=6),
                        dbc.Col(labeled_input("Nacionalidad", dcc.Dropdown(
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
                    labeled_input("Municipio de Residencia", dcc.Dropdown(
                        id="in-mcpio-residencia", options=mcpio_options,
                        value=20001, clearable=False,
                    )),
                ]),

                section_card("Contexto Familiar", "bi-house", [
                    dbc.Row([
                        dbc.Col(labeled_input("Estrato de Vivienda", dcc.Dropdown(
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
                        dbc.Col(labeled_input("Personas en el Hogar", dcc.Dropdown(
                            id="in-personas",
                            options=[
                                {"label": "1 a 2",    "value": "1 a 2"},
                                {"label": "3 a 4",    "value": "3 a 4"},
                                {"label": "5 a 6",    "value": "5 a 6"},
                                {"label": "7 a 8",    "value": "7 a 8"},
                                {"label": "9 o más",  "value": "9 o más"},
                                {"label": "12 o más", "value": "12 o más"},
                            ],
                            value="3 a 4", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(labeled_input("Cuartos en el Hogar", dcc.Dropdown(
                            id="in-cuartos",
                            options=[
                                {"label": "1 cuarto",   "value": "1"},
                                {"label": "2 cuartos",  "value": "2"},
                                {"label": "3 cuartos",  "value": "3"},
                                {"label": "4 cuartos",  "value": "4"},
                                {"label": "5 cuartos",  "value": "5"},
                                {"label": "6 a 9",      "value": "6+"},
                                {"label": "10 o más",   "value": "10+"},
                            ],
                            value="3", clearable=False,
                        )), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(labeled_input("Educación de la Madre",
                            dcc.Dropdown(id="in-edu-madre", options=educacion_options,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                        dbc.Col(labeled_input("Educación del Padre",
                            dcc.Dropdown(id="in-edu-padre", options=educacion_options,
                                         value="Secundaria (Bachillerato) completa",
                                         clearable=False)), md=6),
                    ]),
                    html.Label("Recursos del hogar", className="form-label fw-semibold mb-2 input-label"),
                    dbc.Row([
                        dbc.Col(dbc.Checklist(
                            id="in-recursos",
                            options=[
                                {"label": "Computador", "value": "computador"},
                                {"label": "Internet",   "value": "internet"},
                                {"label": "Automóvil",  "value": "automovil"},
                                {"label": "Lavadora",   "value": "lavadora"},
                            ],
                            value=["computador", "internet"],
                            switch=True,
                            inline=False,
                        )),
                    ]),
                ]),

                section_card("Contexto Escolar", "bi-building", [
                    labeled_input("Municipio del Colegio", dcc.Dropdown(
                        id="in-mcpio-cole", options=mcpio_options,
                        value=20001, clearable=False,
                    )),
                    labeled_input("Colegio", dcc.Dropdown(
                        id="in-colegio",
                        options=[],
                        placeholder="Seleccione el municipio primero…",
                        clearable=True,
                    )),
                    dbc.Alert(
                        [html.I(className="bi bi-magic me-2"),
                         "Al seleccionar el colegio, los campos de abajo se completan automáticamente. "
                         "Puedes ajustarlos manualmente."],
                        color="light", className="py-1 px-2 mb-3 small banner-alert",
                        dismissable=False,
                    ),
                    dbc.Row([
                        dbc.Col(labeled_input("Área", dcc.Dropdown(
                            id="in-area",
                            options=[{"label": "Urbano", "value": True},
                                     {"label": "Rural",  "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(labeled_input("Calendario", dcc.Dropdown(
                            id="in-calendario",
                            options=[{"label": "Calendario A", "value": True},
                                     {"label": "Calendario B", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                        dbc.Col(labeled_input("Naturaleza", dcc.Dropdown(
                            id="in-naturaleza",
                            options=[{"label": "Oficial (Pública)", "value": True},
                                     {"label": "No Oficial (Privada)", "value": False}],
                            value=True, clearable=False,
                        )), md=4),
                    ]),
                    dbc.Row([
                        dbc.Col(labeled_input("Jornada", dcc.Dropdown(
                            id="in-jornada",
                            options=[
                                {"label": "Única",    "value": "UNICA"},
                                {"label": "Mañana",   "value": "MAÑANA"},
                                {"label": "Tarde",    "value": "TARDE"},
                                {"label": "Completa", "value": "COMPLETA"},
                                {"label": "Sabatina", "value": "SABATINA"},
                                {"label": "Noche",    "value": "NOCHE"},
                            ],
                            value="UNICA", clearable=False,
                        )), md=4),
                        dbc.Col(labeled_input("Carácter", dcc.Dropdown(
                            id="in-caracter",
                            options=[
                                {"label": "Académico",           "value": "ACADÉMICO"},
                                {"label": "Técnico",             "value": "TÉCNICO"},
                                {"label": "Técnico / Académico", "value": "TÉCNICO/ACADÉMICO"},
                            ],
                            value="ACADÉMICO", clearable=False,
                        )), md=4),
                        dbc.Col(labeled_input("Bilingüe", dcc.Dropdown(
                            id="in-bilingue",
                            options=[{"label": "No", "value": False},
                                     {"label": "Sí", "value": True}],
                            value=False, clearable=False,
                        )), md=4),
                    ]),
                ]),

                # Predict button
                dbc.Button(
                    [html.I(className="bi bi-graph-up-arrow me-2"), "Predecir Puntaje"],
                    id="btn-predict", size="lg",
                    className="w-100 mb-2 fw-bold predict-btn",
                ),
                html.Small(
                    "La predicción es una estimación basada en datos históricos del Cesar.",
                    className="text-muted d-block text-center mb-4",
                ),

            ], md=5),

            # ── RIGHT: Results panel ─────────────────────────────────────────
            dbc.Col([

                html.Div(id="result-panel", children=[
                    html.Div([
                        html.Div(
                            html.I(className="bi bi-clipboard-data",
                                   style={"fontSize": "3.5rem", "color": CREMA2}),
                            className="text-center mb-3",
                        ),
                        html.H5("Complete el formulario y presione Predecir",
                                className="text-center text-muted"),
                        html.P(
                            "El modelo estimará el puntaje global esperado en la Prueba Saber 11.",
                            className="text-center text-muted small",
                        ),
                    ], className="py-5 px-3"),
                ]),

                # How to interpret
                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-info-circle me-2"), "Cómo interpretar el resultado"],
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
                                html.Span(f"{lo}–{hi}: {lbl}", className="small"),
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
                        html.P([
                            html.Strong("Rango de confianza: "),
                            f"±{info['metrics']['mae']:.0f} pts (MAE del modelo).",
                        ], className="small mb-1"),
                        html.P([
                            html.Strong("Referencia histórica: "),
                            f"Promedio Cesar = {info['target_mean']:.0f} pts.",
                        ], className="small mb-0"),
                    ], className="py-2 px-3"),
                ], className="mb-3 shadow-sm"),

                # Model metrics card
                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-cpu me-2"), "Métricas del Modelo"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(f"{info['metrics']['mae']:.1f}",
                                         className="metric-value", style={"color": "#c0392b"}),
                                html.Div("MAE (pts)", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div(f"{info['metrics']['rmse']:.1f}",
                                         className="metric-value", style={"color": "#d35400"}),
                                html.Div("RMSE (pts)", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div(f"{info['metrics']['r2']:.3f}",
                                         className="metric-value", style={"color": "#1e8449"}),
                                html.Div("R²", className="metric-label"),
                            ], className="text-center"),
                            dbc.Col([
                                html.Div(f"{info['n_features']}",
                                         className="metric-value", style={"color": VINOTINTO}),
                                html.Div("Variables", className="metric-label"),
                            ], className="text-center"),
                        ]),
                    ], className="py-2"),
                ], className="mb-3 shadow-sm"),

                # Global SHAP card (always visible)
                dbc.Card([
                    dbc.CardHeader(html.H6(
                        [html.I(className="bi bi-bar-chart-line me-2"),
                         "Importancia Global de Variables (SHAP)"],
                        className="mb-0 fw-bold section-title",
                    ), className="section-header"),
                    dbc.CardBody([
                        html.P(
                            "Media del valor absoluto SHAP por grupo de variables, "
                            "calculada sobre 300 estudiantes del conjunto de prueba. "
                            "Refleja cuántos puntos aporta en promedio cada grupo a la predicción.",
                            className="small text-muted mb-2",
                        ),
                        dcc.Graph(
                            figure=make_global_shap_chart(),
                            config={"displayModeBar": False},
                        ),
                    ], className="py-2 px-3"),
                ], className="shadow-sm"),

            ], md=7),
        ]),
    ]),

    # ── Footer ──────────────────────────────────────────────────────────────
    html.Hr(className="mt-4"),
    html.Div([
        html.Small([
            "Desarrollado para el ",
            html.Strong("Ministerio de Educación Nacional"),
            " · Red Neuronal arch_D_deep · Datos ICFES Saber 11, Departamento del Cesar · ",
            html.Span("Universidad de los Andes · IIND 4130", className="text-muted"),
        ]),
    ], className="text-center py-2 text-muted"),

], fluid=True, className="px-4")


# ── Callback: Predict ──────────────────────────────────────────────────────────
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
def predict(
    n_clicks,
    periodo, genero, nacionalidad, mcpio_res,
    estrato, personas, cuartos, edu_madre, edu_padre, recursos,
    mcpio_cole, area, calendario, naturaleza, jornada, caracter, bilingue,
    colegio,
):
    if not n_clicks:
        return no_update

    recursos = recursos or []

    user_inputs = {
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
        score = loader.predict(user_inputs)
        score = float(np.clip(score, 0, 500))
    except Exception as e:
        return dbc.Alert(f"Error en la predicción: {e}", color="danger")

    label, color = score_level(score)
    mae        = info["metrics"]["mae"]
    mean_score = info["target_mean"]
    delta_mean = score - mean_score

    delta_text = (
        f"+{delta_mean:.0f} pts por encima del promedio histórico ({mean_score:.0f} pts)"
        if delta_mean >= 0 else
        f"{delta_mean:.0f} pts por debajo del promedio histórico ({mean_score:.0f} pts)"
    )
    delta_color = "#1e8449" if delta_mean >= 0 else "#c0392b"

    # Model-derived group ablation insights
    try:
        group_impacts = loader.explain_groups(user_inputs, top_n=5)
    except Exception:
        group_impacts = []

    def impact_item(label_text: str, impact: float):
        if impact >= 0:
            icon, icon_color, sign = "bi-arrow-up-circle-fill", "#1e8449", f"+{impact:.1f}"
        else:
            icon, icon_color, sign = "bi-arrow-down-circle-fill", "#c0392b", f"{impact:.1f}"
        return html.Li([
            html.I(className=f"bi {icon} me-2", style={"color": icon_color}),
            html.Strong(f"{sign} pts"),
            html.Span(f" · {label_text}", className="text-muted"),
        ], className="mb-1 small")

    if group_impacts:
        insights_content = html.Ul(
            [impact_item(g["label"], g["impact"]) for g in group_impacts],
            className="mb-0 ps-2",
        )
    else:
        insights_content = html.P(
            "No se identificaron factores destacables con esta combinación.",
            className="text-muted small mb-0",
        )

    return [
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(figure=make_gauge(score), config={"displayModeBar": False}),
                className="py-2",
            ),
        ], className="mb-3 shadow"),

        dbc.Card([
            dbc.CardHeader(html.H6(
                [html.I(className="bi bi-pin-map me-2"), "Posición en la escala Saber 11"],
                className="mb-0 fw-bold section-title",
            ), className="section-header"),
            dbc.CardBody([
                dcc.Graph(figure=make_context_bar(score), config={"displayModeBar": False}),
                html.P(delta_text, className="text-center small mt-1",
                       style={"color": delta_color}),
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
                        html.Div(f"{score:.0f}", className="big-score",
                                 style={"color": color}),
                        html.Div("puntos", className="text-muted small"),
                    ], className="text-center", md=3),
                    dbc.Col([
                        html.P([html.Strong("Nivel: "), label], className="mb-1"),
                        html.P([
                            html.Strong("Rango probable: "),
                            f"{max(0, score - mae):.0f} – {min(500, score + mae):.0f} pts",
                        ], className="mb-1 small text-muted"),
                        html.P([
                            html.Strong("Comparado con el Cesar: "),
                            delta_text.split(":")[0] if ":" in delta_text else delta_text,
                        ], className="mb-0 small text-muted"),
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
                html.P(
                    "Impacto estimado de cada factor en tu puntaje vs. el estudiante promedio del Cesar:",
                    className="small text-muted mb-2",
                ),
                insights_content,
            ], className="py-2 px-3"),
        ], className="shadow-sm"),
    ]


# ── Callback: populate school dropdown when municipio changes ──────────────────
@app.callback(
    Output("in-colegio", "options"),
    Output("in-colegio", "value"),
    Input("in-mcpio-cole", "value"),
)
def update_school_options(municipio):
    if not municipio:
        return [], None
    schools = COLEGIOS_BY_MCPIO.get(int(municipio), [])
    options = sorted(
        [{"label": s["nombre"].title(), "value": s["code"]} for s in schools],
        key=lambda x: x["label"],
    )
    return options, None


# ── Callback: autofill school attributes when a colegio is selected ────────────
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
def autofill_school(colegio_code):
    if not colegio_code:
        return no_update, no_update, no_update, no_update, no_update, no_update
    info = COLEGIOS.get(str(colegio_code))
    if not info:
        return no_update, no_update, no_update, no_update, no_update, no_update
    return (
        info["area_urbano"],
        info["bilingue"],
        info["calendario_a"],
        info["oficial"],
        info["caracter"],
        info["jornada"],
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
