import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

print('Cargando datos...')
df = pd.read_csv('data/filtered_icfes_data_cesar.csv')
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Binary encoding
binary_map = {
    'cole_area_ubicacion':  {'URBANO': 1., 'RURAL': 0.},
    'cole_bilingue':        {'S': 1., 'N': 0.},
    'cole_calendario':      {'A': 1., 'B': 0.},
    'cole_naturaleza':      {'OFICIAL': 1., 'NO OFICIAL': 0.},
    'estu_genero':          {'M': 1., 'F': 0.},
    'fami_tieneautomovil':  {'Si': 1., 'No': 0.},
    'fami_tienecomputador': {'Si': 1., 'No': 0.},
    'fami_tieneinternet':   {'Si': 1., 'No': 0.},
    'fami_tienelavadora':   {'Si': 1., 'No': 0.},
}
for col, mapping in binary_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Ordinal encoding
edu_ord = {'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
           'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4,
           'Técnica o tecnológica incompleta': 5, 'Técnica o tecnológica completa': 6,
           'Educación profesional incompleta': 7, 'Educación profesional completa': 8, 'Postgrado': 9}
est_ord = {f'Estrato {i}': i for i in range(1, 7)}
cuartos_ord = {'Uno': 1, 'Dos': 2, 'Tres': 3, 'Cuatro': 4, 'Cinco': 5,
               'Seis': 6, 'Seis o mas': 6, 'Siete': 7, 'Ocho': 8, 'Nueve': 9, 'Diez o más': 10}
personas_ord = {'Una': 1, 'Dos': 2, 'Tres': 3, 'Cuatro': 4, 'Cinco': 5,
                'Seis': 6, 'Siete': 7, 'Ocho': 8, 'Nueve': 9, 'Diez': 10,
                'Once': 11, 'Doce o más': 12}

df['fami_educacionmadre_ord']  = df['fami_educacionmadre'].map(edu_ord).fillna(0).astype('float32')
df['fami_educacionpadre_ord']  = df['fami_educacionpadre'].map(edu_ord).fillna(0).astype('float32')
df['fami_estratovivienda_ord'] = df['fami_estratovivienda'].map(est_ord).fillna(0).astype('float32')
df['fami_cuartoshogar_ord']    = df['fami_cuartoshogar'].map(cuartos_ord).fillna(1).astype('float32')
df['fami_personashogar_ord']   = df['fami_personashogar'].map(personas_ord).fillna(1).astype('float32')

scalar_cols = ['periodo', 'cole_area_ubicacion', 'cole_bilingue', 'cole_calendario',
               'cole_naturaleza', 'estu_genero', 'fami_tieneautomovil', 'fami_tienecomputador',
               'fami_tieneinternet', 'fami_tienelavadora', 'fami_educacionmadre_ord',
               'fami_educacionpadre_ord', 'fami_estratovivienda_ord',
               'fami_cuartoshogar_ord', 'fami_personashogar_ord']
ohe_cols = ['cole_caracter', 'cole_cod_dane_establecimiento', 'cole_cod_dane_sede',
            'cole_cod_mcpio_ubicacion', 'cole_jornada', 'estu_cod_reside_mcpio', 'estu_nacionalidad']

available_scalar = [c for c in scalar_cols if c in df.columns]
available_ohe    = [c for c in ohe_cols if c in df.columns]

df_model = df[available_scalar + available_ohe + ['punt_global']].copy()
df_model = pd.get_dummies(df_model, columns=available_ohe, drop_first=True)
df_model = df_model.dropna()

X = df_model.drop(columns=['punt_global'])
y = df_model['punt_global'].astype('float32')

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32),
    test_size=0.2, random_state=42
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

print('Entrenando HistGradientBoostingRegressor...')
model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    l2_regularization=0.1, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f'MAE={mae:.4f}  R²={r2:.4f}')

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.2, s=6, color='#1a3a6b', label='Predicciones')

lim_min = min(float(y_test.min()), float(y_pred.min()))
lim_max = max(float(y_test.max()), float(y_pred.max()))
ax.plot([lim_min, lim_max], [lim_min, lim_max], color='red', lw=2, label='Prediccion perfecta')

ax.set_xlabel('Puntaje Real')
ax.set_ylabel('Prediccion')
ax.set_title('Prediccion vs Puntaje Real')
ax.legend()
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data_science/grafico_q1.png', dpi=300,
            bbox_inches='tight', transparent=True, facecolor='white')
plt.show()
print('Guardado: data_science/grafico_q1.png')
