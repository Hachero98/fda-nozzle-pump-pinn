# Mesh-Free Inverse Reconstruction of the Incompressible Navier–Stokes State from Sparse and Direction-Blind PIV Measurements

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Flax_NNX-orange.svg)](https://github.com/google/jax)

A physics-informed neural network (PINN) framework that reconstructs the full
incompressible Navier–Stokes state — velocity, pressure, and wall shear stress
— from sparse experimental particle image velocimetry (PIV) measurements,
validated on both U.S. FDA inter-laboratory benchmark medical devices.

**Status:** Manuscript prepared for submission to *Journal of Computational
Physics*.

## Headline results

- **Hard-constraint magnitude enforcement** — output-transform parametrisation
  $\hat{u} = |\hat{V}|\cos\hat{\theta}$, $\hat{v} = |\hat{V}|\sin\hat{\theta}$
  that enforces PIV magnitude exactly at data points, leaving the Navier–Stokes
  residuals as the sole driver of direction recovery.
- **Mechanism isolation** — four-configuration ablation demonstrates that
  momentum residuals (not continuity, not data fitting alone) are necessary and
  sufficient for direction recovery from direction-blind PIV magnitude.
- **Numerical-uniqueness evidence** — 8-seed deep-ensemble construction with
  pairwise cosine similarity > 0.99 between all recovered vector fields.
- **Cross-dataset transferability** — identical architecture across two FDA
  benchmarks, three orders of magnitude in Reynolds number, two distinct
  geometries.
- **Beats best CFD group** in the Stewart 2012 FDA 28-laboratory study at
  $\text{Re}=500$: PINN $E_g=0.137$ vs best CFD (laminar) $E_g=0.34$
  (2.5× smaller).
- **$<1.7\%$ relative-$L^2$** velocity-magnitude error on all five
  Blood Pump diffuser operating conditions.
- **11–26 minutes** per case on a free Colab T4 GPU; **5–30× speed-up** over
  traditional CFD on the same benchmarks.

## Repository structure

```
.
├── B4_FDA_Nozzle_PINN_Paper.tex            # Manuscript (LaTeX, single file)
├── COVER_LETTER_JCP.md                     # JCP submission cover letter
├── HIGHLIGHTS.txt                          # 5 Elsevier-format highlights
├── refs.bib                                # Bibliography
│
├── fda_nozzle_PINN.py                      # Axisymmetric nozzle PINN (Re 500-3500)
├── fda_pump_diffuser_PINN.py               # 2D pump diffuser PINN (5 conditions)
├── parse_fda_piv.py                        # Hariharan 2011 PIV parser
├── parse_pump_diffuser_piv.py              # Hariharan 2018 PIV parser
├── ablation_pump_C5.py                     # Standalone ablation (local CPU)
├── cfd_vs_pinn_comparison.py               # PINN vs Blom 2023 CFD comparison
│
├── B4_FDA_Nozzle_PINN_Colab.ipynb          # GPU notebook: nozzle training
├── B4_FDA_Pump_Diffuser_PINN_Colab.ipynb   # GPU notebook: pump training
├── B4_JCP_Strengthening_Colab.ipynb        # GPU notebook: 4 JCP-strengthening experiments
│
├── figures/                                # Source PNGs referenced by manuscript
├── submission_tiffs/                       # 600-dpi LZW TIFFs for journal submission
├── B4_results (3)/                         # Final nozzle results (metrics + PIV + figures)
├── B4_pump_results_1/                      # Final pump results (metrics + figures)
└── PUMP_NOTES.md                           # Pump benchmark operating conditions reference
```

## Reproducing the results

All training runs on the free-tier NVIDIA T4 GPU available through Google Colab.

### FDA nozzle benchmark (Re = 500, 2000, 3500)

[Open in Colab](https://colab.research.google.com/github/Hachero98/fda-nozzle-pump-pinn/blob/main/B4_FDA_Nozzle_PINN_Colab.ipynb)

1. Open the notebook above in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run Cell 2 (install), then Runtime → Restart runtime
4. Runtime → Run all
5. ~45 minutes total wall-clock; auto-downloads `B4_results.zip`

### FDA Blood Pump diffuser benchmark (5 operating conditions)

[Open in Colab](https://colab.research.google.com/github/Hachero98/fda-nozzle-pump-pinn/blob/main/B4_FDA_Pump_Diffuser_PINN_Colab.ipynb)

Same workflow as the nozzle. Requires uploading the 5 parsed PIV CSVs when
prompted. ~55 minutes wall-clock; auto-downloads `B4_pump_results.zip`.

### JCP-strengthening experiments (ablation, hard-constraint, UQ, HP sensitivity)

[Open in Colab](https://colab.research.google.com/github/Hachero98/fda-nozzle-pump-pinn/blob/main/B4_JCP_Strengthening_Colab.ipynb)

Bundles all four methodological-strengthening experiments into one Colab
session: 4-configuration ablation, hard-constraint architectural variant,
8-seed deep-ensemble UQ, and 3×3 hyperparameter-sensitivity grid. ~3–4 hours
wall-clock on T4; auto-downloads `B4_jcp_strengthening.zip`.

### Local CPU runs

For smoke testing without GPU:

```bash
pip install jax flax optax jaxopt numpy scipy matplotlib openpyxl pyvista
python fda_pump_diffuser_PINN.py --case C5 --quick
python ablation_pump_C5.py --quick
```

`--quick` mode uses reduced iteration counts (1500–5000 Adam vs 30,000–60,000
in publication mode) and completes in 1–5 minutes per case on CPU.

## Datasets

Both benchmark datasets are publicly available from the FDA Office of Science
and Engineering Laboratories (OSEL) GitHub repository:

- **FDA Critical Path nozzle PIV** (Hariharan et al., *J. Biomech. Eng.* 2011):
  `OSEL-DAM/CFD-and-Blood-Damage-Benchmarks/Nozzle/Data/SE_exp_*.zip`
- **FDA Blood Pump diffuser PIV** (Hariharan et al., *Cardiovasc. Eng.
  Technol.* 2018): `OSEL-DAM/CFD-and-Blood-Damage-Benchmarks/Blood
  Pump/Data/Diffuser/*.xlsx`

Repository: <https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks>

## Citation

If you use this code or build on the methodology, please cite both the
software (this repository) and the paper:

```bibtex
@software{Hackman2026FDA_PINN_Code,
  author  = {Hackman, Emmanuel},
  title   = {Mesh-Free Inverse Reconstruction of the Incompressible
             Navier-Stokes State from Sparse and Direction-Blind PIV
             Measurements (code)},
  year    = {2026},
  url     = {https://github.com/Hachero98/fda-nozzle-pump-pinn},
  note    = {Reproducibility code for the FDA-nozzle and Blood-Pump PINN paper}
}

@article{Hackman2026FDA_PINN_Paper,
  author  = {Hackman, Emmanuel},
  title   = {Mesh-Free Inverse Reconstruction of the Incompressible
             Navier-Stokes State from Sparse and Direction-Blind PIV
             Measurements: A Physics-Informed Neural Network Framework
             Validated on the FDA Critical Path Nozzle and Blood Pump
             Benchmarks},
  journal = {Journal of Computational Physics},
  year    = {2026},
  note    = {Under review}
}
```

GitHub displays a "Cite this repository" button generated from
[`CITATION.cff`](CITATION.cff) on the repository landing page.

## License

Source code in this repository is released under the
[Apache License 2.0](LICENSE). The manuscript prose, figures, and
benchmarking analyses are © 2026 Emmanuel Hackman, all rights reserved
pending journal publication.

The FDA benchmark datasets used in this work are public-domain U.S.
Government works released by FDA OSEL and are not redistributed here.

## Acknowledgements

The author thanks the U.S. FDA Office of Science and Engineering
Laboratories for releasing the Critical Path nozzle and Blood Pump
benchmark datasets, and the Google Colab team for the free-tier GPU
infrastructure on which all training experiments were performed.

## Contact

Emmanuel Hackman
PhD Student, Computational Science
University of Southern Mississippi
<emmanuelhackman825@gmail.com>
