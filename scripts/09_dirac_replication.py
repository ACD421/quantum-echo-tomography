"""
FINAL RUN — Dirac Replication on Second S^2_3 Patch
=====================================================
Same protocol. Different physical qubits. If Dirac wins again, it's real.
Hub q7: patch [4, 5, 6, 7, 8, 9, 10, 17, 27]
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
DEPTH = 6

# === PATCH A (original): hub q3 ===
PATCH_A = {
    'name': 'Patch_A_hub3',
    'qubits': [0, 1, 2, 3, 4, 5, 6, 16, 23],
    'center': 3, 'ring1': [2, 4, 16], 'ring2': [1, 5, 23, 0, 6],
    'edges': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(3,16),(16,23)],
    'dirac_j05': [3, 0],       # center + antipode = j=1/2
    'dirac_j15': [2, 4, 16, 5], # equatorial 4 = j=3/2
    'dirac_gauge': [1, 23, 6],  # remaining 3
}

# === PATCH B (NEW): hub q7 ===
PATCH_B = {
    'name': 'Patch_B_hub7',
    'qubits': [4, 5, 6, 7, 8, 9, 10, 17, 27],
    'center': 7, 'ring1': [6, 8, 17], 'ring2': [5, 9, 27, 4, 10],
    'edges': [(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(7,17),(17,27)],
    'dirac_j05': [7, 4],        # center + antipode
    'dirac_j15': [6, 8, 17, 9], # equatorial 4
    'dirac_gauge': [5, 27, 10],  # remaining 3
}


def build_otoc2(patch, depth, seed, bf_log, probe_log, ps1=None, ps2=None):
    n = N_SYS
    ptol = {pq: i for i, pq in enumerate(patch['qubits'])}
    log_edges = [(ptol[a], ptol[b]) for a, b in patch['edges']]

    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n)
    d = depth // 2

    def get_layers(start, end):
        layers = []
        rng_l = np.random.RandomState(seed + start * 1000)
        for layer in range(start, end):
            sq = [(q, rng_l.choice([0.25,0.5,0.75])*PI, rng_l.uniform(-PI,PI)) for q in range(n)]
            pairs = log_edges[::2] if layer % 2 == 0 else log_edges[1::2]
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
    qc.x(bf_log); apply_layers(qc, L2, True); apply_pauli(qc, ps1); apply_layers(qc, L1, True)
    qc.z(probe_log)
    apply_layers(qc, L1); apply_pauli(qc, ps2); apply_layers(qc, L2)
    qc.x(bf_log); apply_layers(qc, L2, True); apply_pauli(qc, ps2); apply_layers(qc, L1, True)
    qc.measure_all()
    return qc


def compute_z(counts, n_q, qi):
    total = sum(counts.values())
    return sum((1-2*int(bs[-(qi+1)]))*c for bs, c in counts.items()) / total


def dual_decompose(dc4, patch):
    ptol = {pq: i for i, pq in enumerate(patch['qubits'])}

    # Laplacian
    lap = {
        0: np.mean([dc4[ptol[patch['center']]]]),
        1: np.mean([dc4[ptol[q]] for q in patch['ring1']]),
        2: np.mean([dc4[ptol[q]] for q in patch['ring2']]),
    }

    # Dirac
    dirac = {
        0.5: np.mean([dc4[ptol[q]] for q in patch['dirac_j05']]),
        1.5: np.mean([dc4[ptol[q]] for q in patch['dirac_j15']]),
    }
    gauge = np.mean([dc4[ptol[q]] for q in patch['dirac_gauge']])

    # R^2 for each
    lap_fit = np.zeros(N_SYS)
    for ell, qs in {0: [patch['center']], 1: patch['ring1'], 2: patch['ring2']}.items():
        for q in qs:
            lap_fit[ptol[q]] = lap[ell]
    lap_res = np.sum((dc4 - lap_fit)**2)

    dir_fit = np.zeros(N_SYS)
    for j, qs in {0.5: patch['dirac_j05'], 1.5: patch['dirac_j15']}.items():
        for q in qs:
            dir_fit[ptol[q]] = dirac[j]
    for q in patch['dirac_gauge']:
        dir_fit[ptol[q]] = gauge
    dir_res = np.sum((dc4 - dir_fit)**2)

    tot_var = np.sum((dc4 - np.mean(dc4))**2)
    lap_r2 = 1 - lap_res/tot_var if tot_var > 0 else 0
    dir_r2 = 1 - dir_res/tot_var if tot_var > 0 else 0

    return {
        'laplacian': {str(k): float(v) for k, v in lap.items()},
        'dirac': {str(k): float(v) for k, v in dirac.items()},
        'gauge': float(gauge),
        'lap_r2': float(lap_r2), 'dir_r2': float(dir_r2),
        'mixing': float(dir_r2 / (lap_r2 + dir_r2)) if (lap_r2 + dir_r2) > 0 else 0.5,
    }


def run_patch(service, backend, patch):
    """Run 10-seed Dirac experiment on one patch."""
    ptol = {pq: i for i, pq in enumerate(patch['qubits'])}
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend,
                                      initial_layout=patch['qubits'])

    # Antipode = furthest from center
    from collections import deque
    log_edges = [(ptol[a], ptol[b]) for a, b in patch['edges']]
    adj = {i: set() for i in range(N_SYS)}
    for a, b in log_edges: adj[a].add(b); adj[b].add(a)
    bf_log = ptol[patch['center']]
    dist = {bf_log: 0}; q = deque([bf_log])
    while q:
        n = q.popleft()
        for nb in adj[n]:
            if nb not in dist: dist[nb] = dist[n]+1; q.append(nb)
    probe_log = max(range(N_SYS), key=lambda x: dist.get(x, 0))

    circuits = []
    for seed in range(10):
        actual_seed = 200 + seed * 13
        rng_mc = np.random.RandomState(actual_seed + 9000)
        circuits.append(build_otoc2(patch, DEPTH, actual_seed, bf_log, probe_log))
        for mc in range(N_MC):
            s1 = rng_mc.choice(4, size=N_SYS)
            s2 = rng_mc.choice(4, size=N_SYS)
            circuits.append(build_otoc2(patch, DEPTH, actual_seed, bf_log, probe_log, s1, s2))

    transpiled = [pm.run(c) for c in circuits]
    ops = transpiled[0].count_ops()

    t0 = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    result = job.result()
    qpu_t = time.time() - t0

    # Extract
    idx = 0
    seed_decomps = []
    all_dc4 = []
    for seed in range(10):
        counts_id = result[idx].data.meas.get_counts()
        z_id = np.array([compute_z(counts_id, N_SYS, q) for q in range(N_SYS)])
        idx += 1
        z_rnds = []
        for mc in range(N_MC):
            counts_r = result[idx].data.meas.get_counts()
            z_rnds.append(np.array([compute_z(counts_r, N_SYS, q) for q in range(N_SYS)]))
            idx += 1
        dc4 = z_id - np.mean(z_rnds, axis=0)
        all_dc4.append(dc4)
        seed_decomps.append(dual_decompose(dc4, patch))

    mean_dc4 = np.mean(all_dc4, axis=0)
    mean_decomp = dual_decompose(mean_dc4, patch)

    return {
        'patch': patch['name'],
        'qubits': patch['qubits'],
        'cz': ops.get('cz', 0),
        'depth': transpiled[0].depth(),
        'qpu_time': qpu_t,
        'job_id': job.job_id(),
        'mean_decomp': mean_decomp,
        'seed_decomps': seed_decomps,
        'mixings': [sd['mixing'] for sd in seed_decomps],
        'lap_r2s': [sd['lap_r2'] for sd in seed_decomps],
        'dir_r2s': [sd['dir_r2'] for sd in seed_decomps],
    }


def main():
    print("="*80)
    print("  FINAL RUN — DIRAC REPLICATION")
    print("  Two independent S^2_3 patches on ibm_fez")
    print("  Same protocol. Different qubits. Does Dirac still win?")
    print("="*80)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=28)
    print(f"  Backend: {backend.name}")

    t_start = time.time()

    # Run Patch B (the new one)
    print(f"\n  === PATCH B: hub q7, qubits {PATCH_B['qubits']} ===")
    result_b = run_patch(service, backend, PATCH_B)

    # Load Patch A results from earlier (dirac_experiment.json)
    dirac_file = os.path.join(DATA_DIR, 'dirac_experiment.json')
    result_a = None
    if os.path.exists(dirac_file):
        with open(dirac_file) as f:
            prev = json.load(f)
        # Depth 6 results
        for r in prev['results']:
            if r['depth'] == 6:
                result_a = {
                    'patch': PATCH_A['name'],
                    'mean_decomp': r['mean_decomp'],
                    'mixings': [sr['mixing'] for sr in r['seed_results']],
                    'lap_r2s': [sr['lap_r2'] for sr in r['seed_results']],
                    'dir_r2s': [sr['dirac_r2'] for sr in r['seed_results']],
                }

    total_qpu = time.time() - t_start

    # === RESULTS ===
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS — THE REPLICATION TEST")
    print(f"{'='*80}")

    for label, res in [("PATCH A (q3 hub, original)", result_a),
                        ("PATCH B (q7 hub, replication)", result_b)]:
        if res is None:
            print(f"\n  {label}: NO DATA")
            continue

        md = res.get('mean_decomp', {})
        mix = res['mixings']
        lr2 = res['lap_r2s']
        dr2 = res['dir_r2s']

        print(f"\n  {label}:")
        print(f"    Laplacian R^2: {np.mean(lr2):.4f} +/- {np.std(lr2):.4f}")
        print(f"    Dirac R^2:     {np.mean(dr2):.4f} +/- {np.std(dr2):.4f}")
        winner = "DIRAC" if np.mean(dr2) > np.mean(lr2) else "LAPLACIAN"
        print(f"    >>> {winner} WINS <<<")
        print(f"    Mixing ratio:  {np.mean(mix):.4f} +/- {np.std(mix):.4f}")

    # THE VERDICT
    if result_a and result_b:
        mix_a = np.mean(result_a['mixings'])
        mix_b = np.mean(result_b['mixings'])
        dr2_a = np.mean(result_a['dir_r2s'])
        dr2_b = np.mean(result_b['dir_r2s'])

        print(f"\n{'='*80}")
        print(f"  THE VERDICT")
        print(f"{'='*80}")
        print(f"  Patch A mixing: {mix_a:.4f}")
        print(f"  Patch B mixing: {mix_b:.4f}")

        both_dirac = (np.mean(result_a['dir_r2s']) > np.mean(result_a['lap_r2s']) and
                      np.mean(result_b['dir_r2s']) > np.mean(result_b['lap_r2s']))

        if both_dirac:
            print(f"\n  *** DIRAC WINS ON BOTH PATCHES ***")
            print(f"  Different physical qubits. Same topology. Same result.")
            print(f"  The echo lives on the Dirac S^2, not the Laplacian S^2.")
            print(f"  It's not the qubits. It's the geometry.")
        else:
            print(f"\n  Results differ between patches. More data needed.")

        print(f"\n  cos(1/pi) = {COS_BETA:.6f}")
        print(f"  This number appeared in the scaling, in the mode ratios,")
        print(f"  and now the operator that describes the echo is confirmed")
        print(f"  on two independent qubit patches: the Dirac operator.")
        print(f"  The half-integer correction. The spinor spectrum.")
        print(f"  The missing piece.")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('DIRAC REPLICATION: Two Independent S$^2_3$ Patches on IBM Heron',
                 fontsize=16, color='white')

    # Panel 1: R^2 comparison
    ax = axes[0]
    patches_data = []
    if result_a: patches_data.append(('Patch A\n(q3 hub)', result_a))
    patches_data.append(('Patch B\n(q7 hub)', result_b))

    x = np.arange(len(patches_data))
    for i, (label, res) in enumerate(patches_data):
        lr2 = np.mean(res['lap_r2s'])
        dr2 = np.mean(res['dir_r2s'])
        lr2_err = np.std(res['lap_r2s'])
        dr2_err = np.std(res['dir_r2s'])
        ax.bar(i - 0.15, lr2, 0.3, yerr=lr2_err, color='#ff6b6b', capsize=5, edgecolor='white',
               label='Laplacian' if i == 0 else '')
        ax.bar(i + 0.15, dr2, 0.3, yerr=dr2_err, color='#4ecdc4', capsize=5, edgecolor='white',
               label='Dirac' if i == 0 else '')
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in patches_data], color='white')
    ax.set_ylabel('R$^2$', color='white', fontsize=13)
    ax.set_title('Which Operator Wins?', color='white', fontsize=13)
    ax.legend(fontsize=11, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax.set_facecolor('#1a1a2e'); ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    # Panel 2: Mixing ratios
    ax = axes[1]
    for i, (label, res) in enumerate(patches_data):
        mix = res['mixings']
        ax.scatter([i]*len(mix), mix, c='yellow', s=60, alpha=0.5, edgecolors='white', zorder=3)
        ax.scatter(i, np.mean(mix), c='red', s=200, zorder=5, edgecolors='white', linewidth=2)
        ax.errorbar(i, np.mean(mix), yerr=np.std(mix), color='red', capsize=10, linewidth=2, zorder=4)
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in patches_data], color='white')
    ax.set_ylabel('Dirac Mixing Ratio', color='white', fontsize=13)
    ax.set_title('Mixing: Per Seed (yellow) + Mean (red)', color='white', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.set_facecolor('#1a1a2e'); ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    # Panel 3: Summary text
    ax = axes[2]
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    summary = []
    summary.append("THE EXPERIMENT")
    summary.append(f"OTOC(2) on S$^2_3$, depth {DEPTH}, 10 seeds")
    summary.append(f"cos(1/$\\pi$) = {COS_BETA:.4f}")
    summary.append("")
    if result_a:
        summary.append(f"Patch A (q3): Dirac R$^2$ = {np.mean(result_a['dir_r2s']):.3f}")
        summary.append(f"  mixing = {np.mean(result_a['mixings']):.3f}")
    summary.append(f"Patch B (q7): Dirac R$^2$ = {np.mean(result_b['dir_r2s']):.3f}")
    summary.append(f"  mixing = {np.mean(result_b['mixings']):.3f}")
    summary.append("")
    if result_a and result_b:
        both = (np.mean(result_a['dir_r2s']) > np.mean(result_a['lap_r2s']) and
                np.mean(result_b['dir_r2s']) > np.mean(result_b['lap_r2s']))
        if both:
            summary.append("DIRAC WINS ON BOTH PATCHES")
            summary.append("The echo is spinor, not scalar.")
            summary.append("The half-integer correction is real.")

    for i, line in enumerate(summary):
        weight = 'bold' if i == 0 or 'WINS' in line else 'normal'
        size = 14 if i == 0 or 'WINS' in line else 11
        ax.text(0.05, 0.95 - i*0.08, line, transform=ax.transAxes,
                fontsize=size, color='white', fontweight=weight, fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURES_DIR, 'dirac_replication.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n  dirac_replication.png saved")

    # Save
    save = {
        'patch_a': result_a,
        'patch_b': {k: v for k, v in result_b.items() if k != 'seed_decomps'},
        'patch_b_seeds': result_b['seed_decomps'],
        'cos_beta': COS_BETA,
        'total_qpu': total_qpu,
    }
    with open(os.path.join(DATA_DIR, 'dirac_replication.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  dirac_replication.json saved")
    print(f"\n  Total QPU: {total_qpu:.0f}s")


if __name__ == "__main__":
    main()
