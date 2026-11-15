"""Figuras analiticas de la revision; no integran nuevas campanas dinamicas."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfcx

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures' / 'revision'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

t = np.linspace(0, 8, 801)
s = 2.0
continued = erfcx(np.sqrt(s+t))
reset = erfcx(math.sqrt(s)) * erfcx(np.sqrt(t))
continued_integer = np.exp(-(s+t))
reset_integer = math.exp(-s) * np.exp(-t)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
axes[0].plot(t, continued, color='#164b73', label='Historia desde 0')
axes[0].plot(t, reset, '--', color='#b35a00', label='Reinicio en s = 2')
axes[0].set(xlabel=r'Tiempo posterior $\tau$', ylabel=r'$x(s+\tau)$',
            title=r'Caputo, $q=1/2$')
axes[0].legend(frameon=False)
axes[1].plot(t, continued_integer, color='#164b73', label='Continuación')
axes[1].plot(t, reset_integer, '--', color='#b35a00', label='Reinicio')
axes[1].set(xlabel=r'Tiempo posterior $\tau$', ylabel=r'$x(s+\tau)$',
            title=r'Control entero, $q=1$')
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=.22)
fig.tight_layout()
fig.savefig(OUT/'reinicio_caputo.pdf')
fig.savefig(OUT/'reinicio_caputo.png', dpi=180)
plt.close(fig)
with (OUT/'reinicio_caputo.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tau','caputo_half_continued','caputo_half_reset','integer_continued','integer_reset'])
    w.writerows(zip(t,continued,reset,continued_integer,reset_integer))

n = np.unique(np.geomspace(10, 20000, 400).astype(int))
upper = -np.expm1(math.log(.05)/n)
fig, ax = plt.subplots(figsize=(7.3, 3.7))
ax.loglog(n, upper, color='#164b73', lw=1.8)
for size in (7200, 8400):
    value = -math.expm1(math.log(.05)/size)
    ax.scatter([size], [value], s=32)
    ax.annotate(f'n={size}: {value:.6f}', (size,value),
                xytext=(-150, 18 if size==7200 else -26), textcoords='offset points',
                arrowprops={'arrowstyle':'-', 'color':'#666666'})
ax.set(xlabel='Número de muestras independientes n',
       ylabel=r'Cota superior $1-0.05^{1/n}$')
ax.grid(which='both', alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'cota_iid_cero_contactos.pdf')
fig.savefig(OUT/'cota_iid_cero_contactos.png', dpi=180)
plt.close(fig)
summary = {
    'scope': 'Curvas analiticas y cota condicional; no son campanas nuevas ni estimaciones aplicadas a mallas deterministas.',
    'caputo_half_solution': 'exp(t)*erfc(sqrt(t)) = erfcx(sqrt(t))',
    'restart_at_s': s,
    'maximum_restart_difference_on_interval': float(np.max(np.abs(continued-reset))),
    'integer_semigroup_residual': float(np.max(np.abs(continued_integer-reset_integer))),
    'iid_confidence_level': .95,
    'iid_upper_bounds': {str(k):-math.expm1(math.log(.05)/k) for k in (7200,8400)},
}
(OUT/'datos_figuras_revision.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert summary['maximum_restart_difference_on_interval'] > .05
assert summary['integer_semigroup_residual'] < 1e-15
print(json.dumps(summary,ensure_ascii=True))
