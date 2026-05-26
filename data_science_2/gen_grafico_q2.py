import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Load data (same preprocessing as Pregunta2.ipynb)
df = pd.read_csv('data/clean_icfes_data_cesar.csv')
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df.columns = df.columns.str.lower()
df = df.dropna(subset=['cole_bilingue'])

features_num = ['punt_lectura_critica', 'punt_matematicas', 'punt_sociales_ciudadanas',
                'punt_c_naturales', 'punt_ingles', 'punt_global', 'fami_estratovivienda']
features_cat = ['cole_naturaleza', 'cole_jornada', 'estu_genero',
                'fami_tieneinternet', 'fami_tienecomputador']

all_features = features_num + features_cat + ['cole_bilingue']
df_clean = df[all_features].copy().dropna()
df_clean['fami_estratovivienda'] = df_clean['fami_estratovivienda'].str.extract(r'(\d+)', expand=False).astype(float)
df_clean = df_clean.dropna(subset=['fami_estratovivienda'])

df_encoded = pd.get_dummies(df_clean, columns=features_cat, drop_first=True, dtype=float)
feature_cols = [c for c in df_encoded.columns
                if c in features_num or any(cat in c for cat in features_cat)]

X = df_encoded[feature_cols].copy()
y = (df_encoded['cole_bilingue'] == 'S').astype(int)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_arr = X_test.astype(float).values

# Load saved model
model = tf.keras.models.load_model('modelos/mejor_modelo_real.keras')
y_pred_proba = model.predict(X_test_arr, verbose=0).flatten()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_val = roc_auc_score(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#1a3a6b', lw=2.5, label=f'ROC (AUC = {auc_val:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Baseline aleatorio')

ax.set_xlabel('Tasa de Falsos Positivos')
ax.set_ylabel('Tasa de Verdaderos Positivos')
ax.set_title('Curva ROC - Clasificacion Bilingue')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(loc='lower right')
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('data_science_2/grafico_q2_roc.png', dpi=300,
            bbox_inches='tight', transparent=True, facecolor='white')
plt.show()
print(f'AUC-ROC: {auc_val:.4f}')
print('Guardado: data_science_2/grafico_q2_roc.png')
