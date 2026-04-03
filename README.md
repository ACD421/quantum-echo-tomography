# Quantum Echo Tomography

Per-qubit OTOC(2) tomography on IBM Heron quantum hardware, with cross-platform comparison against Google Willow circuit data.

**Author:** Andrew C. Dorman  
**Date:** April 2, 2026  
**Hardware:** IBM ibm_fez (Heron r2, 156 qubits), raw SamplerV2 (no error mitigation)  
**Backend for Google extraction:** cirq simulator on Google's published Zenodo circuits (DOI: 10.5281/zenodo.15640503)

## What This Is

The first per-qubit spatial decomposition of quantum scrambling echoes on hardware. Standard OTOC experiments report a single scalar value. This work measures the full spatial echo landscape across all qubits, decomposes it into spherical harmonic modes on 9-qubit S^2_3 fuzzy sphere patches, and tests which operator (Laplacian vs Dirac) better describes the mode structure.

## Key Findings

### What's solid

- **Per-qubit OTOC(2) tomography** on quantum hardware. Nobody has published this before. The echo has genuine, reproducible spatial structure — not just a scalar.
- **Spherical harmonic decomposition** on S^2_3 patches embedded in IBM's heavy-hex topology. 9-qubit patches with 1+3+3+2 distance structure from degree-3 hubs, grouped as 1+3+5 for Laplacian modes.
- **Three independent patches** on ibm_fez (hubs q3, q7, q151) all show the same qualitative behavior: the Laplacian's 1+3+5 decomposition captures zero variance (R^2 ~ 0.008) at deep scrambling, while a 2+4+3 partition captures ~25% of variance. Reproduced on different physical qubits.
- **Google's hidden spatial echo** extracted for the first time. Their published 18-qubit circuits (Nature, Oct 2025) contain full per-qubit information that was never reported. The corrected extraction shows the expected scrambling gradient: strongest signal at the butterfly qubit, decaying across the lattice, inverting at the probe.
- **Cross-platform comparison**: IBM Heron (heavy-hex) and Google Willow (square lattice) show qualitatively similar echo scaling with system size.

### What's not solid

- **The Dirac operator claim does not hold up.** A permutation test (14_robustness_tests.py) shows that the specific j=1/2, j=3/2, gauge qubit assignments are not significantly better than random 2+4+3 partitions (p = 0.44 on Patch A). The R^2 advantage over the Laplacian comes from group size balance (2+4+3 is more balanced than 1+3+5), not from spinor physics. The scripts and data for the Dirac analysis are included for transparency.
- **Mode amplitude ratios** are noisy and patch-dependent (j=3/2/j=1/2 ratios of 3.1, 6.2, 1.7 across three patches). No universal constant emerges.
- **The breathing ratio** |l=2|/|l=1| touches cos(1/pi) at depth 6 but diverges at other depths. One data point, not a plateau.

## Experiments

| Script | Experiment | QPU Time | Backend |
|--------|-----------|----------|---------|
| 01 | Ideal simulator baseline (4-12 qubits) | 0 | Statevector |
| 02 | 10-qubit QPU baseline | 8s | ibm_fez |
| 03 | Size + geometry options (10/16/20q, 12q x3) | 56s | ibm_kingston |
| 04 | Full size sweep (10-156 qubits) | 283s | ibm_fez |
| 05 | S^2_3 sphere at 5 depths | 112s | ibm_fez |
| 06 | Z/X/Y Pauli tomography on S^2_3 | 68s | ibm_fez |
| 07 | Butterfly-averaged sphere (9 seeds) | 77s | ibm_fez |
| 08 | Dirac vs Laplacian (10 seeds x 2 depths) | ~100s | ibm_fez |
| 09 | Dirac replication (Patch B, 10 seeds) | 78s | ibm_fez |
| 10 | Third patch (Patch C, 5 seeds) | 42s | ibm_fez |
| 11 | Google 18q circuit extraction (50 circuits x 21 MC) | 0 (sim) | cirq |
| 12 | Google data analysis + cross-platform comparison | 0 | N/A |
| 13 | Bitstring state visualization | 0 | N/A |
| 14 | Robustness tests (permutation, distance, effect size) | 0 | N/A |

Total QPU: ~550 seconds on IBM Quantum free tier.

## Known Issues

- **Scripts 04 (sweep)**: System sizes 20, 36, 70, 100 produce trivial circuits after transpilation (depth 17, ~15-81 CZ gates). These data points are invalid — the RNG produces unitaries where U and U-dagger partially cancel at those sizes. Valid data points: 10, 16, 27, 50, 130, 156 qubits.
- **Butterfly placement confound**: In all sphere experiments, the butterfly (X perturbation) is placed at the hub qubit. This confounds the Dirac decomposition analysis. Resolving this requires running the butterfly at a non-hub position (not done due to QPU budget).
- **No error mitigation**: All QPU results are raw SamplerV2 shots. No dynamical decoupling, no twirling, no readout correction. The DC4 estimator (identity minus random mean) provides some noise cancellation but does not correct systematic errors.

## Setup

```bash
pip install -r requirements.txt
export IBM_QUANTUM_TOKEN=your_token_here
```

For Google extraction (scripts 11-12), download circuit data from [Zenodo](https://doi.org/10.5281/zenodo.15640503) and place in `google_circuits/OTOC2_circuits/`.

## Data

All experimental results are in `data/` as JSON files. All figures are in `figures/` as PNG files. These are the actual outputs from the April 2, 2026 experimental runs — not regenerated.

## Job IDs (IBM Quantum, verifiable)

- Patch B replication: `d77hs5bc6das739gpjvg`
- Patch C: `d77in7oeecps73d64hug`
- Additional job IDs are recorded in the JSON data files.

## Citation

If you use this work, please cite:
```
Dorman, A.C. (2026). Quantum Echo Tomography: Per-Qubit OTOC(2) Spatial 
Decomposition on IBM Heron. GitHub: ACD421/quantum-echo-tomography.
```

## License

Proprietary. See LICENSE. Academic citation permitted.
