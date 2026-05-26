import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Results from Pregunta_3.ipynb Phase 1 (already executed)
phase1_results = {
    'arch_1': {'mae_mean': 6.9795, 'r2_mean': 0.2715},
    'arch_2': {'mae_mean': 6.9035, 'r2_mean': 0.2878},
    'arch_3': {'mae_mean': 6.9592, 'r2_mean': 0.2743},
    'arch_4': {'mae_mean': 6.8677, 'r2_mean': 0.2905},
    'arch_5': {'mae_mean': 6.9384, 'r2_mean': 0.2790},
}

arch_names = list(phase1_results.keys())
mae_values = [phase1_results[a]['mae_mean'] for a in arch_names]
best_idx   = mae_values.index(min(mae_values))

colors = ['#e07b00' if i == best_idx else '#1a3a6b' for i in range(len(arch_names))]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(arch_names, mae_values, color=colors, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Arquitectura')
ax.set_ylabel('MAE Promedio')
ax.set_title('Comparacion de Arquitecturas')
ax.set_ylim(6.82, 7.02)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [Patch(facecolor='#e07b00', label='Mejor arquitectura'),
                   Patch(facecolor='#1a3a6b', label='Otras arquitecturas')]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('data_science_3/grafico_q3_arquitecturas.png', dpi=300,
            bbox_inches='tight', transparent=True, facecolor='white')
plt.show()
print(f'Mejor: {arch_names[best_idx]}  MAE={mae_values[best_idx]:.4f}')
print('Guardado: data_science_3/grafico_q3_arquitecturas.png')
