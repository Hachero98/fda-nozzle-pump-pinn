# Cover Letter — Journal of Computational Physics

**To:** The Editors, *Journal of Computational Physics*
**From:** Emmanuel Hackman, School of Mathematics and Natural Sciences, University of Southern Mississippi
**Manuscript title:** Mesh-Free Inverse Reconstruction of the Incompressible Navier–Stokes State from Sparse and Direction-Blind PIV Measurements: A Physics-Informed Neural Network Framework Validated on the FDA Critical Path Nozzle and Blood Pump Benchmarks

---

Dear Editors,

I am submitting the enclosed manuscript for consideration as a research article in the *Journal of Computational Physics*. The work develops a mesh-free physics-informed neural network (PINN) framework for the inverse reconstruction of the full incompressible Navier–Stokes state — velocity, pressure, and wall shear stress — from sparse experimental particle image velocimetry (PIV) measurements, and validates the methodology against both inter-laboratory benchmark medical devices released by the U.S. Food and Drug Administration's Office of Science and Engineering Laboratories.

### Methodological contributions

Two contributions distinguish this work from prior PINN literature on fluid mechanics, including the foundational papers of Raissi, Perdikaris, and Karniadakis (*J.\ Comput.\ Phys.* 378, 2019; *J.\ Comput.\ Phys.* 406, 2020 — Hidden Fluid Mechanics) on which our formulation builds:

1. **Direction recovery from direction-blind PIV.** We demonstrate that the Navier–Stokes residuals contain sufficient information to reconstruct a fully vector-resolved velocity field from PIV measurements that report only the velocity magnitude $|V| = \sqrt{u^2 + v^2}$. The momentum and continuity residuals act as a self-consistency constraint that selects the unique physically-realisable vector field from the equivalence class compatible with the scalar observation. To our knowledge no prior PINN-for-fluids study has exploited this mechanism: existing work either uses fully-resolved vector PIV (Hariharan 2011, Raissi 2020) or uses passive-scalar concentration data (Raissi 2020 HFM) where the direction problem does not arise. On the FDA Blood Pump diffuser, where only $|V|$ is available, this mechanism yields relative-$L^2$ error below $1.7\%$ at every operating condition spanning $\mathrm{Re}_{\text{pump}} = 2.1$–$2.9 \times 10^5$.

2. **Cross-dataset transferability of a single PINN methodology.** We show that an identical architecture (6 hidden layers × 64 units, $\tanh$ activation, hard input normalisation to $[-1,+1]^d$), training schedule (cosine-decayed Adam plus L-BFGS finisher), and loss formulation generalises across two FDA benchmark devices, three orders of magnitude in Reynolds number ($\mathrm{Re} \in \{500, 2000, 3500, 2.1\times 10^5, 2.9\times 10^5\}$), two geometric formulations (axisymmetric vs.\ 2D Cartesian), and two PIV data modalities (vector components vs.\ magnitude only). The only modification between applications is the dimensionality of the domain and the data-loss formulation; no architectural or hyperparameter retuning is required.

### Quantitative comparison against established CFD baselines

The manuscript provides a direct comparison against the FDA's 28-laboratory inter-laboratory CFD study (Stewart et al., *Cardiovascular Engineering and Technology* 3, 2012) using its own published global error metric $E_g$. At $\mathrm{Re} = 500$ on the Critical Path nozzle, the PINN attains $E_g = 0.137$, which is $2.5\times$ smaller than the best-performing CFD group (laminar, $E_g = 0.34$) and substantially smaller than every RANS turbulence-model family that participated. This is, to our knowledge, the first direct quantitative comparison of a PINN-based reconstruction against a published inter-laboratory CFD baseline on a benchmark of this rigour.

### Reproducibility

The complete training pipeline — dataset parsers, PINN implementation in JAX/Flax with double-precision arithmetic, two GPU-ready Colab notebooks (one per benchmark), and per-condition metrics — will be released alongside the manuscript on Zenodo and GitHub. All training was performed on the free-tier NVIDIA T4 GPU available through Google Colab, with end-to-end wall-clock times of $11$–$26$~minutes per case. The benchmark datasets (Hariharan 2011, Hariharan 2018) are publicly available from the FDA OSEL GitHub repository.

### Fit with the journal

The work is centred on a methodological advance in physics-informed inverse PDE solvers applied to a problem of direct relevance to the *J.\ Comput.\ Phys.* readership. The PINN community's foundational papers were published in this journal, and we expect the methodological contribution — direction recovery via Navier–Stokes residuals from scalar observation data — to be of interest to researchers working on inverse problems, mesh-free methods, and data-driven scientific computing more broadly.

### Statements

- This manuscript has not been published, nor is it under consideration at any other journal.
- All authors approve the submission.
- The author declares no competing interests.
- This work received no external funding.
- An AI-use disclosure consistent with USM and Elsevier policies is included in the Acknowledgements section.

I would suggest the following potential reviewers (none of whom have collaborated with me in the past three years): G. E. Karniadakis (Brown University), P. Perdikaris (University of Pennsylvania), A. Arzani (University of Utah), L. Lu (Yale University). I have no requested-excluded reviewers.

I would be grateful for your consideration of the manuscript.

Sincerely,

Emmanuel Hackman
PhD Student, Computational Science
School of Mathematics and Natural Sciences
University of Southern Mississippi
Hattiesburg, MS 39406
emmanuelhackman825@gmail.com
