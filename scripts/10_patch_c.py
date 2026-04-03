"""
PATCH C — Third independent S^2_3 patch on ibm_fez
====================================================
5 seeds, depth 6. Answer: is the factor-of-2 in the ratio systematic?
Finds a hub far from patches A (q0-q6) and B (q4-q10) automatically.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import time
import json
import os
from collections import deque

import matplotlib
matplotlib.use('Agg')

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
SHOTS = 4096
N_MC = 5
N_SYS = 9
DEPTH = 6
N_SEEDS = 5  # lean — 38s budget


def find_patch_c(backend):
    """Find a degree-3 hub far from patches A/B, build 9-qubit S^2 patch.
    Heavy-hex gives 1+3+3+2 by BFS distance. We combine d=2+d=3 into ring2
    to get the same 1+3+5 structure as patches A and B."""
    used = {0,1,2,3,4,5,6,7,8,9,10,16,17,23,27}
    cmap = backend.coupling_map.get_edges()

    adj = {}
    for a, b in cmap:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    all_qubits = sorted(adj.keys())
    hubs = [(q, min(abs(q - u) for u in used)) for q in all_qubits
            if len(adj[q]) >= 3 and q not in used]
    hubs.sort(key=lambda x: -x[1])

    for hub, gap in hubs:
        dist = {hub: 0}; queue = deque([hub])
        while queue:
            n = queue.popleft()
            for nb in adj[n]:
                if nb not in dist: dist[nb] = dist[n]+1; queue.append(nb)

        d1 = sorted([q for q in all_qubits if dist.get(q)==1 and q not in used])
        d2 = sorted([q for q in all_qubits if dist.get(q)==2 and q not in used])
        d3 = sorted([q for q in all_qubits if dist.get(q)==3 and q not in used])

        # Need: 3 at d=1, and at least 5 total at d=2+d=3
        if len(d1) >= 3 and len(d2) + len(d3) >= 5:
            r1 = d1[:3]
            r2 = (d2 + d3)[:5]  # combine d=2 and d=3 into ring2
            patch_qubits = [hub] + r1 + r2

            pset = set(patch_qubits)
            edges = sorted(set((min(a,b),max(a,b)) for a,b in cmap if a in pset and b in pset))

            if len(edges) >= 7:
                # Antipode = furthest qubit in the patch
                pdist = {q: dist[q] for q in patch_qubits}
                antipode = max(patch_qubits, key=lambda q: pdist[q])

                # Dirac: j=1/2 = center+antipode, j=3/2 = 4 equatorial, gauge = 3 remaining
                dirac_j05 = [hub, antipode]
                remaining = [q for q in patch_qubits if q not in dirac_j05]
                remaining.sort(key=lambda q: pdist[q])
                dirac_j15 = remaining[:4]
                dirac_gauge = remaining[4:]

                print(f"  Selected hub q{hub} (gap={gap})")
                print(f"  Distance structure: d0=1, d1={len(d1)}, d2={len(d2)}, d3={len(d3)}")
                return {
                    'name': f'Patch_C_hub{hub}',
                    'qubits': patch_qubits,
                    'center': hub,
                    'ring1': r1,
                    'ring2': r2,
                    'edges': edges,
                    'dirac_j05': dirac_j05,
                    'dirac_j15': dirac_j15,
                    'dirac_gauge': dirac_gauge,
                }

    return None


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
    lap = {
        0: np.mean([dc4[ptol[patch['center']]]]),
        1: np.mean([dc4[ptol[q]] for q in patch['ring1']]),
        2: np.mean([dc4[ptol[q]] for q in patch['ring2']]),
    }
    dirac = {
        0.5: np.mean([dc4[ptol[q]] for q in patch['dirac_j05']]),
        1.5: np.mean([dc4[ptol[q]] for q in patch['dirac_j15']]),
    }
    gauge = np.mean([dc4[ptol[q]] for q in patch['dirac_gauge']])

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


def main():
    print("="*80)
    print("  PATCH C — THIRD INDEPENDENT S^2_3 ON ibm_fez")
    print("  5 seeds, depth 6. The factor-of-2 test.")
    print("="*80)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=28)
    print(f"  Backend: {backend.name}")

    # Find patch C automatically
    print("\n  Finding Patch C (far from A and B)...")
    patch = find_patch_c(backend)
    if patch is None:
        print("  ERROR: Could not find suitable patch C")
        return

    print(f"  Patch C: hub q{patch['center']}, qubits {patch['qubits']}")
    print(f"  Edges: {patch['edges']}")
    print(f"  Dirac j=1/2: {patch['dirac_j05']}")
    print(f"  Dirac j=3/2: {patch['dirac_j15']}")
    print(f"  Gauge:        {patch['dirac_gauge']}")

    ptol = {pq: i for i, pq in enumerate(patch['qubits'])}
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend,
                                      initial_layout=patch['qubits'])

    # BFS for probe (antipode)
    log_edges = [(ptol[a], ptol[b]) for a, b in patch['edges']]
    adj_l = {i: set() for i in range(N_SYS)}
    for a, b in log_edges: adj_l[a].add(b); adj_l[b].add(a)
    bf_log = ptol[patch['center']]
    dist = {bf_log: 0}; q = deque([bf_log])
    while q:
        n = q.popleft()
        for nb in adj_l[n]:
            if nb not in dist: dist[nb] = dist[n]+1; q.append(nb)
    probe_log = max(range(N_SYS), key=lambda x: dist.get(x, 0))
    print(f"  Butterfly: logical q{bf_log} (phys q{patch['center']})")
    print(f"  Probe:     logical q{probe_log} (phys q{patch['qubits'][probe_log]})")

    # Build circuits: 5 seeds x (1 identity + 5 MC) = 30 circuits
    circuits = []
    for seed in range(N_SEEDS):
        actual_seed = 500 + seed * 17  # different seed series from A/B
        rng_mc = np.random.RandomState(actual_seed + 9000)
        circuits.append(build_otoc2(patch, DEPTH, actual_seed, bf_log, probe_log))
        for mc in range(N_MC):
            s1 = rng_mc.choice(4, size=N_SYS)
            s2 = rng_mc.choice(4, size=N_SYS)
            circuits.append(build_otoc2(patch, DEPTH, actual_seed, bf_log, probe_log, s1, s2))

    transpiled = [pm.run(c) for c in circuits]
    ops = transpiled[0].count_ops()
    print(f"\n  {len(transpiled)} circuits, ~{ops.get('cz',0)} CZ/circuit, depth ~{transpiled[0].depth()}")
    print(f"  Submitting to {backend.name}...")

    t0 = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    result = job.result()
    qpu_t = time.time() - t0
    print(f"  Job {job.job_id()} done in {qpu_t:.1f}s")

    # Extract
    idx = 0
    seed_decomps = []
    all_dc4 = []
    for seed in range(N_SEEDS):
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

    # === RESULTS ===
    print(f"\n{'='*80}")
    print(f"  PATCH C RESULTS — hub q{patch['center']}")
    print(f"{'='*80}")

    lr2s = [sd['lap_r2'] for sd in seed_decomps]
    dr2s = [sd['dir_r2'] for sd in seed_decomps]
    mixings = [sd['mixing'] for sd in seed_decomps]

    print(f"\n  Laplacian R^2: {np.mean(lr2s):.4f} +/- {np.std(lr2s):.4f}")
    print(f"  Dirac R^2:     {np.mean(dr2s):.4f} +/- {np.std(dr2s):.4f}")
    winner = "DIRAC" if np.mean(dr2s) > np.mean(lr2s) else "LAPLACIAN"
    print(f"  >>> {winner} WINS <<<")
    print(f"  Mixing:        {np.mean(mixings):.4f} +/- {np.std(mixings):.4f}")

    # Mode values
    j05 = mean_decomp['dirac']['0.5']
    j15 = mean_decomp['dirac']['1.5']
    print(f"\n  Dirac modes:")
    print(f"    j=1/2: {j05:.4f}")
    print(f"    j=3/2: {j15:.4f}")
    if abs(j05) > 1e-6:
        raw_ratio = j15 / j05
        print(f"    Raw ratio j=3/2 / j=1/2: {raw_ratio:.3f}")
        print(f"    Degeneracy-normalized:    {raw_ratio/2:.3f}")
    else:
        print(f"    j=1/2 too close to zero for ratio")

    print(f"\n  Laplacian modes:")
    for ell in ['0', '1', '2']:
        print(f"    l={ell}: {mean_decomp['laplacian'][ell]:.4f}")

    # Per-seed ratios
    print(f"\n  Per-seed j=3/2 / j=1/2 ratios:")
    for i, sd in enumerate(seed_decomps):
        j05_s = sd['dirac']['0.5']
        j15_s = sd['dirac']['1.5']
        if abs(j05_s) > 1e-4:
            r = j15_s / j05_s
            print(f"    Seed {i}: {r:.3f} (normalized: {r/2:.3f})")
        else:
            print(f"    Seed {i}: j=1/2 ~ 0, ratio undefined")

    # Compare all three patches
    print(f"\n{'='*80}")
    print(f"  THREE-PATCH COMPARISON")
    print(f"{'='*80}")
    print(f"  Patch A (hub q3):  raw ratio 3.1, normalized 1.55")
    print(f"  Patch B (hub q7):  raw ratio 6.2, normalized 3.1")
    if abs(j05) > 1e-6:
        print(f"  Patch C (hub q{patch['center']}): raw ratio {j15/j05:.1f}, normalized {j15/j05/2:.2f}")
    print(f"\n  Factor-of-2 test: is the scaling systematic?")

    print(f"\n  QPU time: {qpu_t:.1f}s")
    print(f"  Job ID: {job.job_id()}")

    # Save
    save = {
        'patch_c': {
            'name': patch['name'],
            'qubits': patch['qubits'],
            'center': patch['center'],
            'edges': patch['edges'],
            'dirac_j05': patch['dirac_j05'],
            'dirac_j15': patch['dirac_j15'],
            'dirac_gauge': patch['dirac_gauge'],
            'cz': ops.get('cz', 0),
            'depth': transpiled[0].depth(),
            'qpu_time': qpu_t,
            'job_id': job.job_id(),
            'mean_decomp': mean_decomp,
            'seed_decomps': seed_decomps,
            'mixings': mixings,
            'lap_r2s': lr2s,
            'dir_r2s': dr2s,
        },
        'all_dc4': [dc.tolist() for dc in all_dc4],
    }
    with open(os.path.join(DATA_DIR, 'patch_c_result.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Saved: patch_c_result.json")


if __name__ == '__main__':
    main()
