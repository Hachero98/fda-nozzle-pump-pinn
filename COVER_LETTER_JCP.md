# Cover Letter — Journal of Computational Physics

**To:** The Editors, *Journal of Computational Physics*
**From:** Emmanuel Hackman, School of Mathematics and Natural Sciences, University of Southern Mississippi
**Manuscript title:** Mesh-Free Inverse Reconstruction of the Incompressible Navier–Stokes State from Sparse and Direction-Blind PIV Measurements: A Physics-Informed Neural Network Framework Validated on the FDA Critical Path Nozzle and Blood Pump Benchmarks

---

Dear Editors,

I am submitting the enclosed manuscript for consideration as a research article in the *Journal of Computational Physics*. The work develops a mesh-free physics-informed neural network (PINN) framework for the inverse reconstruction of the full incompressible Navier–Stokes state — velocity, pressure, and wall shear stress — from sparse and direction-blind experimental particle image velocimetry (PIV) measurements, and validates the methodology against both inter-laboratory benchmark medical devices released by the U.S. Food and Drug Administration's Office of Science and Engineering Laboratories.

### Methodological contributions

Four contributions distinguish this work from prior PINN literature on fluid mechanics, including the foundational papers of Raissi, Perdikaris, and Karniadakis (*J. Comput. Phys.* 378, 2019; *J. Comput. Phys.* 406, 2020 — Hidden Fluid Mechanics) on which our formulation builds. We benchmark against the FDA's two inter-laboratory CFD studies (Stewart et al., *Cardiovascular Engineering and Technology* 3, 2012, for the nozzle; Ponnaluri et al., *Annals of Biomedical Engineering* 51, 2023, for the blood pump):

1. **Hard-constraint magnitude enforcement (Sec. 4.6).** We introduce an output-transform parametrisation $\hat{u} = |\hat{V}|\cos\hat{\theta}$, $\hat{v} = |\hat{V}|\sin\hat{\theta}$ that enforces the PIV magnitude as a structural hard constraint at every data point, leaving the Navier–Stokes residuals as the *sole* driver of direction recovery. This eliminates the gradient competition between data and physics losses documented by Wang et al. (2021) as a primary failure mode in soft-constraint PINNs. To our knowledge no prior PINN-for-fluids work has used hard-constraint magnitude enforcement for direction-blind inverse problems.

2. **Mechanism isolation: momentum residuals drive direction recovery (Sec. 6.2).** We present a four-configuration ablation that pinpoints *which* component of the composite PINN loss is responsible for the direction-from-magnitude property. The result is sharp and somewhat counterintuitive: the *momentum* residuals, not the continuity constraint and not the data loss in isolation, are necessary and sufficient. Continuity alone (configuration c) preserves the magnitude match but fails to recover the diffuser flow direction. This is a falsifiable mechanism claim, not a black-box observation, and (to our knowledge) the first such ablation for a PINN inverse problem of this type.

3. **Numerical-uniqueness evidence via deep ensembles (Sec. 6.4).** We construct an 8-seed deep-ensemble (Lakshminarayanan et al. 2017) and report pairwise cosine similarities between all recovered vector fields, finding off-diagonal minimum above 0.99 on the pump-diffuser benchmark. This provides empirical evidence for numerical uniqueness of the magnitude-only direction-recovery problem in the absence of a formal theorem, while simultaneously yielding posterior-like per-point uncertainty quantification on velocity, pressure, and wall shear stress.

4. **Cross-dataset transferability across two FDA benchmarks (Secs. 5–6).** A single architecture (6×64 tanh MLP), training schedule (cosine-decayed Adam plus L-BFGS finisher), and loss formulation generalise across two FDA benchmark devices, three orders of magnitude in Reynolds number ($\mathrm{Re} \in \{500, 2000, 3500, 2.1\times 10^5, 2.9\times 10^5\}$), two geometric formulations (axisymmetric vs. 2D Cartesian), and two PIV data modalities (vector components vs. magnitude only). The only modification between applications is the dimensionality of the domain and the data-loss formulation; no architectural or hyperparameter retuning is required, as confirmed by an explicit $3 \times 3$ hyperparameter-sensitivity grid (Sec. 6.5).

### Quantitative comparison against established CFD baselines

The manuscript provides a direct comparison against the FDA's 28-laboratory inter-laboratory CFD study (Stewart et al., *Cardiovascular Engineering and Technology* 3, 2012) using its own published global error metric $E_g$. At $\mathrm{Re} = 500$ on the Critical Path nozzle, the PINN attains $E_g = 0.137$, which is $2.5\times$ smaller than the best-performing CFD group (laminar, $E_g = 0.34$) and substantially smaller than every RANS turbulence-model family that participated. This is, to our knowledge, the first direct quantitative comparison of a PINN-based reconstruction against a published inter-laboratory CFD baseline on a benchmark of this rigour.

### Reproducibility

The complete pipeline — dataset parsers, PINN implementation in JAX/Flax with double-precision arithmetic, three GPU-ready Colab notebooks (main training for each benchmark plus all strengthening experiments: ablation, hard-constraint variant, deep-ensemble UQ, hyperparameter sensitivity), and per-experiment metrics — will be released alongside the manuscript on Zenodo and GitHub. All training was performed on the free-tier NVIDIA T4 GPU available through Google Colab, with end-to-end wall-clock times of $11$–$26$ minutes per main training case and $3$–$4$ hours for the full strengthening-experiment suite. The benchmark datasets (Hariharan 2011, Hariharan 2018) are publicly available from the FDA OSEL GitHub repository.

### Fit with the journal

The work is centred on a methodological advance in physics-informed inverse PDE solvers applied to a problem of direct relevance to the *J.\ Comput.\ Phys.* readership. The PINN community's foundational papers were published in this journal, and we expect the four methodological contributions — hard-constraint magnitude enforcement, mechanism isolation via ablation, deep-ensemble numerical-uniqueness evidence, and demonstrated cross-dataset transferability — to be of interest to researchers working on inverse problems, mesh-free methods, and data-driven scientific computing more broadly.

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
