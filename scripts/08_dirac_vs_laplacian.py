"""
THE DIRAC EXPERIMENT — Dual-Spectrum OTOC(2) on S^2_3
======================================================
10 random seeds x 2 depths x 6 circuits = 120 circuits
Analyze through BOTH Laplacian AND Dirac spectra.
Find the mixing ratio. That's where the shape is.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import time
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN")
if not IBM_TOKEN:
    raise ValueError("Set IBM_QUANTUM_TOKEN environment variable")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PI = math.pi
COS_BETA = math.cos(1/PI)
SHOTS = 4096
N_MC = 5
N_SYS = 9

# S^2_3 on ibm_fez
SPHERE_QUBITS = [0, 1, 2, 3, 4, 5, 6, 16, 23]
CENTER = 3; RING1 = [2, 4, 16]; RING2 = [1, 5, 23, 0, 6]
HW_EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(3,16),(16,23)]
PHYS_TO_LOG = {pq: i for i, pq in enumerate(SPHERE_QUBITS)}
LOG_EDGES = [(PHYS_TO_LOG[a], PHYS_TO_LOG[b]) for a, b in HW_EDGES]

# MODE ASSIGNMENTS
# Laplacian: l=0 (1), l=1 (3), l=2 (5) = 9
LAPLACIAN = {
    0: [PHYS_TO_LOG[CENTER]],                           # l=0: center
    1: [PHYS_TO_LOG[q] for q in RING1],                 # l=1: ring-1
    2: [PHYS_TO_LOG[q] for q in RING2],                 # l=2: ring-2
}

# Dirac: j=1/2 (2), j=3/2 (4) = 6 spinor states
# j=1/2: the two "poles" — center + antipode (spin up/down at the pole)
# j=3/2: the four "equatorial" qubits — ring-1 + closest ring-2 member
# Remaining 3: scalar (gauge) sector
DIRAC = {
    0.5: [PHYS_TO_LOG[CENTER], PHYS_TO_LOG[0]],        # j=1/2: pole pair
    1.5: [PHYS_TO_LOG[2], PHYS_TO_LOG[4],              # j=3/2: equatorial 4
          PHYS_TO_LOG[16], PHYS_TO_LOG[5]],
}
DIRAC_GAUGE = [PHYS_TO_LOG[1], PHYS_TO_LOG[23], PHYS_TO_LOG[6]]  # remaining 3


def build_sphere_otoc2(depth, seed, butterfly_log, probe_log, ps1=None, ps2=None):
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(N_SYS)
    d = depth // 2

    def get_layers(start, end):
        layers = []
        rng_l = np.random.RandomState(seed + start * 1000)
        for layer in range(start, end):
            sq = [(q, rng_l.choice([0.25,0.5,0.75])*PI, rng_l.uniform(-PI,PI)) for q in range(N_SYS)]
            pairs = LOG_EDGES[::2] if layer % 2 == 0 else LOG_EDGES[1::2]
            layers.append({'sq': sq, 'pairs': pairs})
        return layers

    def apply_layers(circ, layers, inv=False):
        if inv:
            for ld in reversed(layers):
                for i,j in ld['pairs']: circ.cz(i,j)
                for q,t,p in ld['sq']: circ.rz(p,q); circ.rx(-t,q); circ.rz(-p,q)
        else:
            for ld in layers:
                for q,t,p in ld['sq']: circ.rz(p,q); circ.rx(t,q); circ.rz(-p,q)
                for i,j in ld['pairs']: circ.cz(i,j)

    def apply_pauli(circ, s):
        if s is None: return
        for q,p in enumerate(s):
            if p==1: circ.x(q)
            elif p==2: circ.y(q)
            elif p==3: circ.z(q)

    L1, L2 = get_layers(0, d), get_layers(d, depth)
    apply_layers(qc, L1); apply_pauli(qc, ps1); apply_layers(qc, L2)
    qc.x(butterfly_log); apply_layers(qc, L2, True); apply_pauli(qc, ps1); apply_layers(qc, L1, True)
    qc.z(probe_log)
    apply_layers(qc, L1); apply_pauli(qc, ps2); apply_layers(qc, L2)
    qc.x(butterfly_log); apply_layers(qc, L2, True); apply_pauli(qc, ps2); apply_layers(qc, L1, True)
    qc.measure_all()
    return qc


def compute_z(counts, n_q, qi):
    total = sum(counts.values())
    return sum((1-2*int(bs[-(qi+1)]))*c for bs, c in counts.items()) / total


def dual_spectrum_decompose(dc4):
    """Decompose echo into both Laplacian and Dirac spectra."""
    # Laplacian
    lap = {}
    for ell, qubits in LAPLACIAN.items():
        lap[ell] = np.mean([dc4[q] for q in qubits])

    # Dirac
    dirac = {}
    for j, qubits in DIRAC.items():
        dirac[j] = np.mean([dc4[q] for q in qubits])
    gauge = np.mean([dc4[q] for q in DIRAC_GAUGE])

    # Mixing ratio: project echo onto Laplacian basis vs Dirac basis
    # Laplacian prediction: echo(q) = sum_l a_l * f_l(q) where f_l = 1 on ring-l
    # Dirac prediction: echo(q) = sum_j b_j * g_j(q) where g_j = 1 on j-sector
    #
    # Compute residuals for each decomposition
    lap_fit = np.zeros(N_SYS)
    for ell, qubits in LAPLACIAN.items():
        for q in qubits:
            lap_fit[q] = lap[ell]
    lap_residual = np.sum((dc4 - lap_fit)**2)

    dirac_fit = np.zeros(N_SYS)
    for j, qubits in DIRAC.items():
        for q in qubits:
            dirac_fit[q] = dirac[j]
    for q in DIRAC_GAUGE:
        dirac_fit[q] = gauge
    dirac_residual = np.sum((dc4 - dirac_fit)**2)

    total_var = np.sum((dc4 - np.mean(dc4))**2)
    lap_r2 = 1 - lap_residual / total_var if total_var > 0 else 0
    dirac_r2 = 1 - dirac_residual / total_var if total_var > 0 else 0

    return {
        'laplacian': lap, 'dirac': dirac, 'gauge': float(gauge),
        'lap_residual': float(lap_residual), 'dirac_residual': float(dirac_residual),
        'lap_r2': float(lap_r2), 'dirac_r2': float(dirac_r2),
        'mixing': float(dirac_r2 / (lap_r2 + dirac_r2)) if (lap_r2 + dirac_r2) > 0 else 0.5,
    }


def main():
    print("="*80)
    print("  THE DIRAC EXPERIMENT")
    print("  Dual-Spectrum OTOC(2) on S^2_3 — Laplacian vs Dirac")
    print("  10 seeds x 2 depths x 6 circuits = finding the mixing ratio")
    print("="*80)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=27)
    print(f"  Backend: {backend.name}")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend,
                                      initial_layout=SPHERE_QUBITS)

    BF_LOG = PHYS_TO_LOG[CENTER]
    PROBE_LOG = PHYS_TO_LOG[0]  # antipode

    all_results = []
    t_start = time.time()

    for depth in [4, 6]:
        seed_results = []
        all_circuits = []
        seed_map = []

        for seed in range(10):
            actual_seed = 100 + seed * 17  # spread seeds
            rng_mc = np.random.RandomState(actual_seed + 5000)

            # Identity
            qc_id = build_sphere_otoc2(depth, actual_seed, BF_LOG, PROBE_LOG)
            all_circuits.append(qc_id)
            seed_map.append((seed, 'id'))

            # MC random
            for mc in range(N_MC):
                s1 = rng_mc.choice(4, size=N_SYS)
                s2 = rng_mc.choice(4, size=N_SYS)
                qc_r = build_sphere_otoc2(depth, actual_seed, BF_LOG, PROBE_LOG, s1, s2)
                all_circuits.append(qc_r)
                seed_map.append((seed, f'rnd_{mc}'))

        print(f"\n  Depth {depth}: {len(all_circuits)} circuits, transpiling...")
        transpiled = [pm.run(c) for c in all_circuits]
        ops = transpiled[0].count_ops()
        print(f"    CZ={ops.get('cz',0)}, depth={transpiled[0].depth()}")

        print(f"    Submitting to {backend.name}...")
        t0 = time.time()
        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=SHOTS)
        result = job.result()
        qpu_t = time.time() - t0
        print(f"    QPU: {qpu_t:.1f}s, job={job.job_id()}")

        # Extract per-seed DC4
        idx = 0
        for seed in range(10):
            counts_id = result[idx].data.meas.get_counts()
            z_id = np.array([compute_z(counts_id, N_SYS, q) for q in range(N_SYS)])
            idx += 1

            z_rnds = []
            for mc in range(N_MC):
                counts_r = result[idx].data.meas.get_counts()
                z_r = np.array([compute_z(counts_r, N_SYS, q) for q in range(N_SYS)])
                z_rnds.append(z_r)
                idx += 1

            dc4 = z_id - np.mean(z_rnds, axis=0)
            decomp = dual_spectrum_decompose(dc4)
            seed_results.append({'seed': seed, 'dc4': dc4.tolist(), **decomp})

        # Aggregate across seeds
        all_dc4 = np.array([sr['dc4'] for sr in seed_results])
        mean_dc4 = all_dc4.mean(axis=0)
        std_dc4 = all_dc4.std(axis=0)

        # Dual decomposition of the MEAN
        mean_decomp = dual_spectrum_decompose(mean_dc4)

        # Per-seed decompositions for error bars
        lap_modes = {ell: [] for ell in LAPLACIAN}
        dirac_modes = {j: [] for j in DIRAC}
        mixings = []
        for sr in seed_results:
            for ell in LAPLACIAN:
                lap_modes[ell].append(sr['laplacian'][ell])
            for j in DIRAC:
                dirac_modes[j].append(sr['dirac'][j])
            mixings.append(sr['mixing'])

        print(f"\n    LAPLACIAN DECOMPOSITION (10 seeds, mean +/- std):")
        for ell in sorted(LAPLACIAN.keys()):
            vals = lap_modes[ell]
            print(f"      l={ell}: {np.mean(vals):+.5f} +/- {np.std(vals):.5f}  (n={len(LAPLACIAN[ell])} qubits)")

        print(f"    DIRAC DECOMPOSITION:")
        for j in sorted(DIRAC.keys()):
            vals = dirac_modes[j]
            print(f"      j={j}: {np.mean(vals):+.5f} +/- {np.std(vals):.5f}  (n={len(DIRAC[j])} qubits)")
        print(f"      gauge: {np.mean([sr['gauge'] for sr in seed_results]):+.5f}")

        # Breathing ratios with error bars
        print(f"\n    BREATHING RATIOS:")
        l0_vals = np.array(lap_modes[0])
        l1_vals = np.array(lap_modes[1])
        l2_vals = np.array(lap_modes[2])

        if np.mean(np.abs(l0_vals)) > 0.001 and np.mean(np.abs(l1_vals)) > 0.001:
            r_l1l0 = np.abs(l1_vals) / (np.abs(l0_vals) + 1e-8)
            print(f"      Laplacian |l1/l0|: {np.mean(r_l1l0):.4f} +/- {np.std(r_l1l0):.4f}")
        if np.mean(np.abs(l1_vals)) > 0.001 and np.mean(np.abs(l2_vals)) > 0.001:
            r_l2l1 = np.abs(l2_vals) / (np.abs(l1_vals) + 1e-8)
            print(f"      Laplacian |l2/l1|: {np.mean(r_l2l1):.4f} +/- {np.std(r_l2l1):.4f}")

        j05_vals = np.array(dirac_modes[0.5])
        j15_vals = np.array(dirac_modes[1.5])
        if np.mean(np.abs(j05_vals)) > 0.001 and np.mean(np.abs(j15_vals)) > 0.001:
            r_dirac = np.abs(j15_vals) / (np.abs(j05_vals) + 1e-8)
            print(f"      Dirac |j=3/2|/|j=1/2|: {np.mean(r_dirac):.4f} +/- {np.std(r_dirac):.4f}")

        print(f"\n    MIXING RATIO (Dirac weight / total):")
        print(f"      {np.mean(mixings):.4f} +/- {np.std(mixings):.4f}")
        print(f"      (0 = pure Laplacian, 1 = pure Dirac, 0.5 = equal)")

        print(f"\n    R^2 — which decomposition fits better?")
        print(f"      Laplacian R^2: {mean_decomp['lap_r2']:.4f}")
        print(f"      Dirac R^2:     {mean_decomp['dirac_r2']:.4f}")
        winner = "DIRAC" if mean_decomp['dirac_r2'] > mean_decomp['lap_r2'] else "LAPLACIAN"
        print(f"      >>> {winner} WINS <<<")

        print(f"    cos(1/pi) = {COS_BETA:.6f}")
        print(f"    cos^2(1/pi) = {COS_BETA**2:.6f}")

        all_results.append({
            'depth': depth, 'qpu_time': qpu_t,
            'mean_dc4': mean_dc4.tolist(), 'std_dc4': std_dc4.tolist(),
            'mean_decomp': mean_decomp,
            'seed_results': seed_results,
            'lap_modes_stats': {str(ell): {'mean': float(np.mean(lap_modes[ell])),
                                           'std': float(np.std(lap_modes[ell]))}
                                for ell in LAPLACIAN},
            'dirac_modes_stats': {str(j): {'mean': float(np.mean(dirac_modes[j])),
                                           'std': float(np.std(dirac_modes[j]))}
                                  for j in DIRAC},
            'mixing_stats': {'mean': float(np.mean(mixings)), 'std': float(np.std(mixings))},
        })

    # =====================================================================
    # PLOT — The Dirac vs Laplacian comparison
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('Dirac vs Laplacian: Which Operator Describes the Echo?',
                 fontsize=16, color='white')

    # Panel 1: Mode amplitudes for both decompositions
    ax = axes[0]
    for ri, r in enumerate(all_results):
        depth = r['depth']
        offset = ri * 0.15

        # Laplacian
        for ell in [0, 1, 2]:
            stats = r['lap_modes_stats'][str(ell)]
            ax.errorbar(ell - 0.1 + offset, stats['mean'], yerr=stats['std'],
                       fmt='o', color=['#ff6b6b', '#ff8888'][ri], markersize=10,
                       capsize=5, label=f'Lap d={depth}' if ell == 0 else '')

        # Dirac
        for j in [0.5, 1.5]:
            stats = r['dirac_modes_stats'][str(j)]
            ax.errorbar(j + 1.5 + offset, stats['mean'], yerr=stats['std'],
                       fmt='D', color=['#4ecdc4', '#88dddd'][ri], markersize=10,
                       capsize=5, label=f'Dirac d={depth}' if j == 0.5 else '')

    ax.axhline(y=0, color='white', alpha=0.3)
    ax.set_xticks([0, 1, 2, 2.0, 3.0])
    ax.set_xticklabels(['$\\ell$=0\n(1q)', '$\\ell$=1\n(3q)', '$\\ell$=2\n(5q)',
                        'j=1/2\n(2q)', 'j=3/2\n(4q)'], color='white', fontsize=10)
    ax.set_ylabel('Mode Amplitude', color='white', fontsize=12)
    ax.set_title('Mode Amplitudes\n(error bars = 10 seeds)', color='white', fontsize=12)
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    # Panel 2: R^2 comparison
    ax = axes[1]
    depths_plot = [r['depth'] for r in all_results]
    lap_r2 = [r['mean_decomp']['lap_r2'] for r in all_results]
    dir_r2 = [r['mean_decomp']['dirac_r2'] for r in all_results]

    x = np.arange(len(depths_plot))
    ax.bar(x - 0.15, lap_r2, 0.3, color='#ff6b6b', label='Laplacian R$^2$', edgecolor='white')
    ax.bar(x + 0.15, dir_r2, 0.3, color='#4ecdc4', label='Dirac R$^2$', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in depths_plot], color='white')
    ax.set_ylabel('R$^2$ (goodness of fit)', color='white', fontsize=12)
    ax.set_title('Which Operator Fits Better?', color='white', fontsize=12)
    ax.legend(fontsize=11, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    # Panel 3: Mixing ratio
    ax = axes[2]
    mix_means = [r['mixing_stats']['mean'] for r in all_results]
    mix_stds = [r['mixing_stats']['std'] for r in all_results]
    ax.errorbar(depths_plot, mix_means, yerr=mix_stds, fmt='o-', color='yellow',
                markersize=15, capsize=8, ecolor='gray', linewidth=2)
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Equal mixing')
    ax.axhline(y=0, color='#ff6b6b', linestyle=':', alpha=0.5, label='Pure Laplacian')
    ax.axhline(y=1, color='#4ecdc4', linestyle=':', alpha=0.5, label='Pure Dirac')
    ax.set_xlabel('Circuit Depth', color='white', fontsize=12)
    ax.set_ylabel('Dirac Mixing Ratio', color='white', fontsize=12)
    ax.set_title('The Shape: Laplacian ↔ Dirac\nMixing Ratio vs Depth', color='white', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURES_DIR, 'dirac_vs_laplacian.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n  dirac_vs_laplacian.png saved")

    # Save
    outfile = os.path.join(DATA_DIR, 'dirac_experiment.json')
    with open(outfile, 'w') as f:
        json.dump({'results': all_results, 'cos_beta': COS_BETA}, f, indent=2, default=str)
    print(f"  dirac_experiment.json saved")
    print(f"\n  Total QPU: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
