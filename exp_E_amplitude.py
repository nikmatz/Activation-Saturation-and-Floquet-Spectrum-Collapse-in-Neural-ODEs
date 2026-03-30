#!/usr/bin/env python3
"""
Illustration E: tanh vs SiLU at increasing orbit amplitude.

Author:  Nikolaos M. Matzakos
         nikmatz@aspete.gr  |  ORCID 0000-0001-8647-6082
License: MIT

Trains both tanh- and SiLU-activated MLPs on the Stuart–Landau
oscillator with increasing limit-cycle radius A.
Standard Xavier init + Adam — no bias tricks.

Demonstrates:
 (a) Training loss diverges for tanh but not SiLU at large A.
 (b) Monodromy determinant drifts toward 1 for tanh, stays near
     the exact value for SiLU.

Output: code/figures/exp_E_amplitude.{pdf,png}
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

# ─── Scaled Stuart–Landau ──────────────────────────────────────────
def target_field(X, A):
    x, y = X[:, 0], X[:, 1]
    r2 = x**2 + y**2
    return np.column_stack([x - y - x * r2 / A**2,
                            x + y - y * r2 / A**2])

# ─── Activations ───────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def act_fwd(a, name):
    if name == 'tanh':
        return np.tanh(a)
    s = _sigmoid(a)
    return a * s          # SiLU

def act_prime(a, name):
    if name == 'tanh':
        return 1.0 - np.tanh(a)**2
    s = _sigmoid(a)
    return s * (1.0 + a * (1.0 - s))

# ─── MLP ───────────────────────────────────────────────────────────
class MLP:
    def __init__(self, H=32, seed=42, act='tanh'):
        self.act = act
        g = np.random.RandomState(seed)
        s1, s2 = np.sqrt(2.0 / (2 + H)), np.sqrt(2.0 / (H + 2))
        self.p = dict(W1=g.randn(H, 2) * s1, b1=np.zeros(H),
                      W2=g.randn(2, H) * s2, b2=np.zeros(2))

    def fwd(self, x):
        a = x @ self.p['W1'].T + self.p['b1']
        return act_fwd(a, self.act) @ self.p['W2'].T + self.p['b2'], a

    def jac_at(self, pt):
        a = self.p['W1'] @ pt + self.p['b1']
        D = act_prime(a, self.act)
        return self.p['W2'] @ np.diag(D) @ self.p['W1'], D

# ─── Adam ──────────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=1e-3):
        self.lr = lr; self.m = {}; self.v = {}; self.t = 0
    def step(self, P, G):
        self.t += 1
        for k in P:
            if k not in self.m:
                self.m[k] = np.zeros_like(P[k])
                self.v[k] = np.zeros_like(P[k])
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * G[k]**2
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

# ─── Training ─────────────────────────────────────────────────────
def train(net, A, epochs=25000, N=2000, lr=2e-3):
    rng = np.random.RandomState(0)
    opt = Adam(lr)
    final = None
    for ep in range(epochs):
        r = rng.uniform(0.1 * A, 2.5 * A, N)
        th = rng.uniform(0, 2 * np.pi, N)
        X = np.column_stack([r * np.cos(th), r * np.sin(th)])
        Y = target_field(X, A)
        pred, a = net.fwd(X)
        res = pred - Y
        loss = np.mean(res**2)
        final = loss
        dL = 2 * res / (N * 2)
        h = act_fwd(a, net.act)
        hp = act_prime(a, net.act)
        dh = dL @ net.p['W2'] * hp
        G = dict(W2=dL.T @ h, b2=dL.sum(0),
                 W1=dh.T @ X, b1=dh.sum(0))
        opt.step(net.p, G)
        if ep % (epochs // 3) == 0 or ep == epochs - 1:
            # Normalised MSE for fair comparison across A
            target_var = np.mean(Y**2)
            nMSE = loss / max(target_var, 1e-8)
            print(f"    {net.act:5s} A={A:4.0f}  ep={ep:5d}  "
                  f"MSE={loss:.3e}  nMSE={nMSE:.3e}")
    return final

# ─── Orbit measurement ────────────────────────────────────────────
def orbit_meas(net, A, n=1000):
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dt = 2 * np.pi / n
    trs = []
    for t in ts:
        pt = np.array([A * np.cos(t), A * np.sin(t)])
        J, _ = net.jac_at(pt)
        trs.append(np.trace(J))
    ln_det = sum(trs) * dt
    return ln_det, np.exp(ln_det)

# ─── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    amplitudes = [1, 4, 16, 64]
    res = {a: [] for a in ('tanh', 'silu')}

    for A in amplitudes:
        for act in ('tanh', 'silu'):
            print(f"\n  ── A = {A},  σ = {act} ──")
            net = MLP(H=32, seed=42, act=act)
            mse = train(net, A, epochs=25000, lr=2e-3)
            ld, dM = orbit_meas(net, A)
            # Normalised MSE
            rng = np.random.RandomState(99)
            rr = rng.uniform(0.1*A, 2.5*A, 5000)
            tth = rng.uniform(0, 2*np.pi, 5000)
            Xv = np.column_stack([rr*np.cos(tth), rr*np.sin(tth)])
            Yv = target_field(Xv, A)
            pv, _ = net.fwd(Xv)
            nmse = np.mean((pv - Yv)**2) / np.mean(Yv**2)

            rec = dict(A=A, mse=mse, nmse=nmse, ln_det=ld, det_M=dM)
            res[act].append(rec)
            print(f"    >> MSE={mse:.3e}  nMSE={nmse:.3e}  "
                  f"ln det M={ld:.3f}  det M={dM:.3e}")

    # ─── Summary ──────────────────────────────────────────────────
    exact = -4 * np.pi
    print(f"\n{'σ':>6} {'A':>4} {'nMSE':>10} {'ln det M':>10} {'det M':>10}")
    print('-' * 50)
    for act in ('tanh', 'silu'):
        for r in res[act]:
            print(f"{act:>6} {r['A']:4d} {r['nmse']:10.3e} "
                  f"{r['ln_det']:10.3f} {r['det_M']:10.3e}")
    print(f"Exact: ln det M = {exact:.3f},  det M = {np.exp(exact):.3e}")

    # ─── Figure ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    As = amplitudes

    # (a) Normalised MSE vs A
    ax1.loglog(As, [r['nmse'] for r in res['tanh']],
               'ro-', ms=9, lw=2, label=r'$\tanh$')
    ax1.loglog(As, [r['nmse'] for r in res['silu']],
               'bs-', ms=9, lw=2, label=r'SiLU')
    ax1.set_xlabel('Orbit amplitude $A$', fontsize=13)
    ax1.set_ylabel('Normalised MSE', fontsize=13)
    ax1.set_title(r'(a) Vector-field fit: $\tanh$ vs.\ SiLU', fontsize=11)
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

    # (b) ln det M vs A
    ax2.semilogx(As, [r['ln_det'] for r in res['tanh']],
                 'ro-', ms=9, lw=2, label=r'$\tanh$')
    ax2.semilogx(As, [r['ln_det'] for r in res['silu']],
                 'bs-', ms=9, lw=2, label=r'SiLU')
    ax2.axhline(exact, color='k', ls='--', alpha=0.5,
                label=r'Exact: $-4\pi$')
    ax2.axhline(0, color='gray', ls=':', alpha=0.3)
    ax2.set_xlabel('Orbit amplitude $A$', fontsize=13)
    ax2.set_ylabel(r'$\ln\det M_\gamma$', fontsize=13)
    ax2.set_title('(b) Monodromy: contraction fidelity', fontsize=11)
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(FIG, f'exp_E_amplitude.{ext}'),
                    bbox_inches='tight', dpi=150)
    print(f"\nFigure saved to {FIG}/exp_E_amplitude.{{pdf,png}}")
