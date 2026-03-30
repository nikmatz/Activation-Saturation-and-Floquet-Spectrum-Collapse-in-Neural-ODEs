# Activation Saturation and Floquet Spectrum Collapse in Neural ODEs

**Author:** Nikolaos M. Matzakos
**Email:** nikmatz@aspete.gr
**ORCID:** [0000-0001-8647-6082](https://orcid.org/0000-0001-8647-6082)
**Affiliation:** School of Pedagogical & Technological Education (ASPETE), Athens, Greece

## About

This repository contains the reproducibility code for the paper:

> N. M. Matzakos, *Activation Saturation and Floquet Spectrum Collapse in Neural ODEs*, arXiv preprint, 2026.

The paper proves that activation saturation imposes a structural dynamical limitation on autonomous Neural ODEs: if *q* hidden layers of the MLP satisfy |σ'| ≤ δ, the input Jacobian is attenuated as ‖Df_θ(x)‖ ≤ C_W · δ^q, forcing every Floquet exponent into the interval [−C_W δ^q, C_W δ^q]. As saturation deepens (δ → 0), the entire Floquet spectrum collapses to zero.

## Repository structure

| File | Description |
|------|-------------|
| `numerical_experiment.py` | **Illustrations A–D.** Jacobian attenuation (Fig. 1), Floquet–Liouville obstruction (Fig. 3), phase portraits (Fig. 2), and Stuart–Landau exact monodromy verification (Table 1). |
| `exp_E_refined_bounds.py` | **Illustration E.** Refined vs original Jacobian bound via saturation-weighted spectral factorisation — Section 6 (Table 3, Fig. 5). |
| `exp_F_individual_multipliers.py` | **Illustration F.** Individual Floquet multiplier bounds — Theorem 4.5 (Table 2, Fig. 4). |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main experiments (Illustrations A–D)
python numerical_experiment.py

# Run refined bound experiment (Illustration E)
python exp_E_refined_bounds.py

# Run individual multiplier bounds (Illustration F)
python exp_F_individual_multipliers.py
```

All figures are saved to `figures/`.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

## Citation

If you use this code, please cite:

```bibtex
@article{matzakos2026saturation,
  title   = {Activation Saturation and Floquet Spectrum Collapse in Neural {ODE}s},
  author  = {Matzakos, Nikolaos M.},
  year    = {2026},
  note    = {arXiv preprint}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
