"""
plot_depth_comparison.py
────────────────────────
Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT
Δείχνει πώς το bound C_W · δ^q σφίγγει εκθετικά με τον αριθμό
κορεσμένων layers q.  Χρησιμοποιεί τα ίδια δεδομένα (δ(s), C_W(s))
από το Illustration A (32-unit MLP, q=1), και υπολογίζει τι θα
συνέβαινε αν είχαμε q=2 ή q=3 κορεσμένα layers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numerical_experiment import TanhMLP, circle_orbit

# ── Εκπαίδευση ίδιου δικτύου (ίδιο seed) ──────────────────────────
SEED = 42
np.random.seed(SEED)
mlp = TanhMLP(n_hidden=32)
mlp.fit(n_iters=7000, lr=0.012, verbose=True)

# ── Υπολογισμός δ(s) και C_W(s) στον μοναδιαίο κύκλο ──────────────
s_values = np.logspace(-0.5, 1.8, 80)
gamma1 = circle_orbit(1.0, 200)

deltas_orbit = []  # max_h δ(h,s) πάνω στον κύκλο
CWs = []

for s in s_values:
    d_max = max(mlp.delta(h, s) for h in gamma1)
    deltas_orbit.append(d_max)
    CWs.append(mlp.CW(s))

deltas_orbit = np.array(deltas_orbit)
CWs = np.array(CWs)

# ── Bounds για q = 1, 2, 3 ─────────────────────────────────────────
q_values = [1, 2, 3]
colors = ['#2176AE', '#E8443A', '#4CAF50']
labels = [r'$q=1$: $C_W \cdot \delta^1$',
          r'$q=2$: $C_W \cdot \delta^2$',
          r'$q=3$: $C_W \cdot \delta^3$']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel (a): Bound vs s ──────────────────────────────────────────
ax = axes[0]
for q, col, lab in zip(q_values, colors, labels):
    bound_q = CWs * deltas_orbit**q
    ax.semilogy(s_values, bound_q, color=col, linewidth=2.2, label=lab)

ax.set_xlabel(r'Pre-activation scale $s$', fontsize=13)
ax.set_ylabel(r'Jacobian bound $C_W \cdot \delta^q$', fontsize=13)
ax.set_title(r'(a) Bound tightens exponentially with depth $q$', fontsize=13)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=1e-30)

# ── Panel (b): δ^q vs δ (θεωρητικό) ────────────────────────────────
ax2 = axes[1]
delta_range = np.linspace(0.01, 1.0, 200)
for q, col, lab in zip(q_values, colors,
                        [r'$\delta^1$', r'$\delta^2$', r'$\delta^3$']):
    ax2.plot(delta_range, delta_range**q, color=col, linewidth=2.2, label=lab)

# Αν ξέρουμε ότι δ=0.4, δείξε τις τιμές
d_example = 0.4
for q, col in zip(q_values, colors):
    ax2.plot(d_example, d_example**q, 'o', color=col, markersize=8, zorder=5)
    ax2.annotate(f'  ${d_example}^{q}={d_example**q:.3f}$',
                 (d_example, d_example**q), fontsize=10, color=col,
                 va='bottom' if q == 1 else 'top')

ax2.set_xlabel(r'Saturation level $\delta$', fontsize=13)
ax2.set_ylabel(r'$\delta^q$', fontsize=13)
ax2.set_title(r'(b) Exponential decay in $q$', fontsize=13)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(figdir, exist_ok=True)
for ext in ('pdf', 'png'):
    p = os.path.join(figdir, f'depth_comparison.{ext}')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    print(f"Saved: {p}")
plt.close()
