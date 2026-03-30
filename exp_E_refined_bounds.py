"""
Illustration E: Refined vs Original Jacobian Bound on Stuart-Landau
===================================================================
Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT

Train a 32-unit tanh MLP at s=1 on Stuart-Landau, freeze weights,
sweep s from 1 to 30.  Compare (§5 of the paper):
  (i)   Actual ||Df_theta(h*)||
  (ii)  Original bound: C(U) = ||W2|| · delta · s · ||W1||
  (iii) Refined bound:  C̃(U) = ||W2 D^{1/2}|| · ||D^{1/2} (sW1)||

Verifies the chain  actual ≤ C̃(U) ≤ C(U)  (Corollary cor:uniform).

Produces a 2-panel figure for the paper.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ── Stuart-Landau vector field ────────────────────────────
def stuart_landau(h):
    x, y = h
    return np.array([x - y - x*(x**2 + y**2),
                     x + y - y*(x**2 + y**2)])

# ── Train a simple 2-layer tanh MLP ──────────────────────
def init_mlp(d_in, d_hidden, d_out):
    W1 = np.random.randn(d_hidden, d_in) * np.sqrt(2.0 / (d_in + d_hidden))
    b1 = np.zeros(d_hidden)
    W2 = np.random.randn(d_out, d_hidden) * np.sqrt(2.0 / (d_hidden + d_out))
    b2 = np.zeros(d_out)
    return W1, b1, W2, b2

def train_mlp(W1, b1, W2, b2, n_iters=7000, lr=1e-3, n_samples=500):
    for it in range(n_iters):
        r = 0.1 + 1.9 * np.random.rand(n_samples)
        theta = 2 * np.pi * np.random.rand(n_samples)
        H = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        targets = np.array([stuart_landau(h) for h in H])

        A = (W1 @ H.T).T + b1
        Z = np.tanh(A)
        pred = (W2 @ Z.T).T + b2

        err = pred - targets
        loss = np.mean(err**2)

        dW2 = (2.0 / n_samples) * err.T @ Z
        db2 = (2.0 / n_samples) * err.sum(axis=0)
        dZ = (2.0 / n_samples) * err @ W2
        dA = dZ * (1 - np.tanh(A)**2)
        dW1 = dA.T @ H
        db1 = dA.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if it % 2000 == 0:
            print(f"  iter {it:5d}  loss = {loss:.6f}")
    print(f"  iter {n_iters:5d}  loss = {loss:.6f}  (final)")
    return W1, b1, W2, b2


def compute_bounds(h, W1, b1, W2, b2, s):
    a = s * (W1 @ h + b1)
    d = 1.0 / np.cosh(a)**2     # sech^2
    D = np.diag(d)
    J = W2 @ D @ (s * W1)

    actual = norm(J, ord=2)
    delta = np.max(d)
    original = norm(W2, ord=2) * delta * s * norm(W1, ord=2)

    D_sqrt = np.diag(np.sqrt(d))
    refined = norm(W2 @ D_sqrt, ord=2) * norm(D_sqrt @ (s * W1), ord=2)

    return actual, original, refined, delta


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
print("Training MLP on Stuart-Landau (n=32, s=1)...")
W1, b1, W2, b2 = init_mlp(2, 32, 2)
W1, b1, W2, b2 = train_mlp(W1, b1, W2, b2, n_iters=7000, lr=1e-3)

# ── Find test point with all pre-activations away from 0 ─
print("\nSelecting test point...")
best_h, best_min = None, 0
for r in np.linspace(0.5, 2.0, 20):
    for th in np.linspace(0, 2*np.pi, 10000, endpoint=False):
        h = np.array([r * np.cos(th), r * np.sin(th)])
        min_abs = np.min(np.abs(W1 @ h + b1))
        if min_abs > best_min:
            best_min = min_abs
            best_h = h.copy()
print(f"  h* = ({best_h[0]:.2f}, {best_h[1]:.2f}), min|pre-act| = {best_min:.4f}")

# ── Sweep s ─────────────────────────────────────────────
s_values = np.linspace(1, 30, 100)
actual_arr, original_arr, refined_arr, delta_arr = [], [], [], []
for s in s_values:
    a, o, r, d = compute_bounds(best_h, W1, b1, W2, b2, s)
    actual_arr.append(a)
    original_arr.append(o)
    refined_arr.append(r)
    delta_arr.append(d)

actual_arr = np.array(actual_arr)
original_arr = np.array(original_arr)
refined_arr = np.array(refined_arr)
delta_arr = np.array(delta_arr)
ratio_arr = original_arr / np.maximum(refined_arr, 1e-30)

# ── Print table for paper ───────────────────────────────
print("\nTable for paper:")
print(f"{'s':>5s} {'Actual':>8s} {'Refined':>8s} {'Original':>8s} {'Ratio':>8s} {'delta':>8s}")
idx = np.linspace(0, len(s_values)-1, 10, dtype=int)
for i in idx:
    print(f"{s_values[i]:5.1f} {actual_arr[i]:8.3f} {refined_arr[i]:8.3f} "
          f"{original_arr[i]:8.3f} {ratio_arr[i]:5.1f}x   {delta_arr[i]:8.3f}")

# Verify ordering
assert np.all(actual_arr <= refined_arr * 1.001), "actual > refined!"
assert np.all(refined_arr <= original_arr * 1.001), "refined > original!"
print("\nBound ordering verified: actual <= C_tilde(U) <= C(U)")

# ── Two-panel figure ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): Three bounds
ax1.semilogy(s_values, original_arr, 'k-', lw=1.8,
             label=r'Original bound $C(U) = \|W_2\|\,\delta\,s\|W_1\|$')
ax1.semilogy(s_values, refined_arr, 'r-', lw=1.8,
             label=r'Refined bound $\widetilde{C}(U) = \|W_2 D^{1/2}\|\cdot\|D^{1/2} sW_1\|$')
ax1.semilogy(s_values, actual_arr, 'b-', lw=1.8,
             label=r'Actual $\|Df_\theta(h^\star)\|$')
ax1.fill_between(s_values, actual_arr, refined_arr,
                 alpha=0.12, color='blue', label='Gap (actual to refined)')
ax1.fill_between(s_values, refined_arr, original_arr,
                 alpha=0.12, color='red', label='Gap (refined to original)')
ax1.set_xlabel(r'Pre-activation scale $s$', fontsize=11)
ax1.set_ylabel(r'Jacobian norm (log scale)', fontsize=11)
ax1.set_title('(a)  Actual norm, refined bound, original bound', fontsize=11)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.grid(True, alpha=0.25, which='both')
ax1.set_xlim(1, 30)

# Panel (b): Improvement ratio
ax2.plot(s_values, ratio_arr, 'g-', lw=2)
ax2.axhline(y=1, color='gray', ls='--', alpha=0.4)
ax2.set_xlabel(r'Pre-activation scale $s$', fontsize=11)
ax2.set_ylabel(r'$\rho = C(U)\,/\,\widetilde{C}(U)$', fontsize=11)
ax2.set_title(r'(b)  Improvement ratio $\rho = C(U)/\widetilde{C}(U)$', fontsize=11)
ax2.grid(True, alpha=0.25)
ax2.set_xlim(1, 30)
ax2.set_ylim(0.8, None)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'exp_E_refined_bounds.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")

# Also save PNG for quick preview
out_png = os.path.join(out_dir, 'exp_E_refined_bounds.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Preview saved to {out_png}")

plt.close()
print("Done.")
