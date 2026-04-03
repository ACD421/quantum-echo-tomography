"""
FULL SEND — Butterfly-Averaged S^2_3 + All Visualizations
==========================================================
1. Run butterfly at ALL 9 positions (Z-basis), 5 MC each → isolate scalar mode
2. Produce publication-quality PNGs from ALL data
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
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

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
SPHERE_QUBITS = [0, 1, 2, 3, 4, 5, 6, 16, 23]
CENTER = 3
RING1 = [2, 4, 16]
RING2 = [1, 5, 23, 0, 6]
HW_EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(3,16),(16,23)]
PHYS_TO_LOG = {pq: i for i, pq in enumerate(SPHERE_QUBITS)}
LOG_EDGES = [(PHYS_TO_LOG[a], PHYS_TO_LOG[b]) for a, b in HW_EDGES]


def build_sphere_otoc2(depth, seed, butterfly_log, probe_log, pauli_s1=None, pauli_s2=None):
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

    # PASS 1
    apply_layers(qc, L1); apply_pauli(qc, pauli_s1); apply_layers(qc, L2)
    qc.x(butterfly_log)
    apply_layers(qc, L2, True); apply_pauli(qc, pauli_s1); apply_layers(qc, L1, True)
    qc.z(probe_log)
    # PASS 2
    apply_layers(qc, L1); apply_pauli(qc, pauli_s2); apply_layers(qc, L2)
    qc.x(butterfly_log)
    apply_layers(qc, L2, True); apply_pauli(qc, pauli_s2); apply_layers(qc, L1, True)

    qc.measure_all()
    return qc


def compute_z(counts, n_q, qi):
    total = sum(counts.values())
    v = sum((1-2*int(bs[-(qi+1)]))*c for bs, c in counts.items())
    return v / total


def run_butterfly_average(service, backend, pm, depth=6):
    """Run OTOC(2) with butterfly at EVERY position, average to get scalar mode."""
    print(f"\n  BUTTERFLY-AVERAGED SPHERE (depth={depth})")
    rng = np.random.RandomState(42)
    all_dc4 = np.zeros((N_SYS, N_SYS))  # [butterfly_pos, qubit] -> DC4

    circuits_all = []
    configs = []

    for bf_phys in SPHERE_QUBITS:
        bf_log = PHYS_TO_LOG[bf_phys]
        # Find probe: furthest qubit from butterfly on the graph
        from collections import deque
        adj = {i: set() for i in range(N_SYS)}
        for a, b in LOG_EDGES:
            adj[a].add(b); adj[b].add(a)
        dist = {bf_log: 0}
        q = deque([bf_log])
        while q:
            n = q.popleft()
            for nb in adj[n]:
                if nb not in dist:
                    dist[nb] = dist[n]+1; q.append(nb)
        probe_log = max(range(N_SYS), key=lambda x: dist.get(x, 0))

        # Identity circuit
        qc_id = build_sphere_otoc2(depth, 42, bf_log, probe_log)
        circuits_all.append(qc_id)
        configs.append(('id', bf_phys, bf_log, probe_log))

        # MC randoms
        for mc in range(N_MC):
            s1 = rng.choice(4, size=N_SYS)
            s2 = rng.choice(4, size=N_SYS)
            qc_r = build_sphere_otoc2(depth, 42, bf_log, probe_log, s1, s2)
            circuits_all.append(qc_r)
            configs.append(('rnd', bf_phys, bf_log, probe_log))

    print(f"    Total circuits: {len(circuits_all)}")
    transpiled = [pm.run(c) for c in circuits_all]

    # Submit in batches to avoid timeout
    batch_size = 54  # 9 butterfly positions × 6 circuits
    all_results = []
    t0 = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    print(f"    Job {job.job_id()} submitted, waiting...")
    result = job.result()
    qpu_time = time.time() - t0
    print(f"    QPU wall time: {qpu_time:.1f}s")

    # Extract per-butterfly DC4
    idx = 0
    for bf_i, bf_phys in enumerate(SPHERE_QUBITS):
        # Identity result
        counts_id = result[idx].data.meas.get_counts()
        z_id = np.array([compute_z(counts_id, N_SYS, q) for q in range(N_SYS)])
        idx += 1

        # MC results
        z_rnds = []
        for mc in range(N_MC):
            counts_r = result[idx].data.meas.get_counts()
            z_r = np.array([compute_z(counts_r, N_SYS, q) for q in range(N_SYS)])
            z_rnds.append(z_r)
            idx += 1

        dc4 = z_id - np.mean(z_rnds, axis=0)
        all_dc4[bf_i] = dc4

    # Average over all butterfly positions → scalar mode
    scalar_echo = np.mean(all_dc4, axis=0)

    # Mode decomposition of the scalar echo
    from collections import deque as _
    center_log = PHYS_TO_LOG[CENTER]
    r1_logs = [PHYS_TO_LOG[q] for q in RING1]
    r2_logs = [PHYS_TO_LOG[q] for q in RING2]

    l0 = scalar_echo[center_log]
    l1 = np.mean([scalar_echo[q] for q in r1_logs])
    l2 = np.mean([scalar_echo[q] for q in r2_logs])

    print(f"\n    BUTTERFLY-AVERAGED SCALAR MODE:")
    for pq in SPHERE_QUBITS:
        lq = PHYS_TO_LOG[pq]
        ring = "l=0" if pq == CENTER else ("l=1" if pq in RING1 else "l=2")
        print(f"      q{pq:>2d} ({ring}): avg DC4 = {scalar_echo[lq]:+.6f}")

    print(f"\n    l=0 (center):  {l0:+.6f}")
    print(f"    l=1 (ring-1):  {l1:+.6f}")
    print(f"    l=2 (ring-2):  {l2:+.6f}")
    if abs(l0) > 0.001:
        r10 = abs(l1/l0)
        print(f"    |l1/l0| = {r10:.6f}  (cos(1/pi) = {COS_BETA:.6f})")
        if abs(l1) > 0.001:
            r21 = abs(l2/l1)
            print(f"    |l2/l1| = {r21:.6f}  (cos(1/pi) = {COS_BETA:.6f})")

    return all_dc4, scalar_echo, {'l0': l0, 'l1': l1, 'l2': l2, 'qpu_time': qpu_time}


def make_plots(all_dc4=None, scalar_echo=None, modes=None):
    """Generate all publication PNGs from saved data."""
    print(f"\n  GENERATING PLOTS...")

    # Load all saved data
    files = {}
    for fn in ['qpu_fullsweep.json', 'listen_echoes.json', 'sphere_s2n3.json']:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            with open(path) as f:
                files[fn] = json.load(f)

    # =====================================================================
    # PLOT 1: Echo Scaling with cos(1/pi) overlay
    # =====================================================================
    if 'qpu_fullsweep.json' in files:
        sweep = files['qpu_fullsweep.json']
        valid = [r for r in sweep['results'] if r['cz_gates'] >= r['n_sys']]

        fig, ax = plt.subplots(figsize=(10, 6))
        ns = [r['n_sys'] for r in valid]
        means = [r['dc4_mean_abs'] for r in valid]

        ax.scatter(ns, means, s=100, c='cyan', edgecolors='white', zorder=5, label='IBM Heron (raw)')

        # cos(1/pi)^N fit overlay
        if len(valid) >= 2:
            n_range = np.linspace(min(ns), max(ns), 100)
            # Fit A * cos(1/pi)^N
            A = means[0] / COS_BETA**ns[0]
            cos_curve = A * COS_BETA**n_range
            ax.plot(n_range, cos_curve, 'r--', linewidth=2, alpha=0.8,
                    label=f'cos(1/$\\pi$)$^N$ fit (A={A:.3f})')

        ax.set_xlabel('System Size (qubits)', fontsize=14)
        ax.set_ylabel('Mean |$\\Delta C^{(4)}$|', fontsize=14)
        ax.set_title('OTOC(2) Echo Scaling — IBM Heron ibm_fez\nBreathing Factor cos(1/$\\pi$) = 0.9498', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'echo_scaling.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    echo_scaling.png")

    # =====================================================================
    # PLOT 2: Sphere Correlation Heatmap
    # =====================================================================
    if 'listen_echoes.json' in files:
        data = files['listen_echoes.json']
        zz = np.array(data['Z_dc4_conn'])

        fig, ax = plt.subplots(figsize=(8, 7))
        labels = [f'q{q}' for q in SPHERE_QUBITS]
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='equal')
        ax.set_xticks(range(9)); ax.set_xticklabels(labels, color='white')
        ax.set_yticks(range(9)); ax.set_yticklabels(labels, color='white')
        plt.colorbar(im, label='ZZ Connected Correlation ($\\Delta C^{(4)}$)')
        ax.set_title('S$^2_3$ Echo Correlation Matrix — 9 Qubits on IBM Heron', fontsize=13, color='white')

        # Mark rings
        for i, pq in enumerate(SPHERE_QUBITS):
            if pq == CENTER:
                ax.annotate('l=0', (i, i), color='yellow', fontsize=8, ha='center', va='center', fontweight='bold')
            elif pq in RING1:
                ax.annotate('l=1', (i, i), color='lime', fontsize=7, ha='center', va='center')

        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0d0d1a')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'sphere_correlation.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    sphere_correlation.png")

    # =====================================================================
    # PLOT 3: Correlation Decay — Sphere vs Flat fit
    # =====================================================================
    if 'listen_echoes.json' in files:
        data = files['listen_echoes.json']
        zz = np.array(data['Z_dc4_conn'])

        # Graph distances
        from collections import deque
        adj = {i: set() for i in range(9)}
        for a, b in LOG_EDGES: adj[a].add(b); adj[b].add(a)
        gdist = np.zeros((9,9))
        for s in range(9):
            visited = {s: 0}; q = deque([s])
            while q:
                n = q.popleft()
                for nb in adj[n]:
                    if nb not in visited: visited[nb] = visited[n]+1; q.append(nb)
            for e, d in visited.items(): gdist[s][e] = d

        dists, corrs = [], []
        for i in range(9):
            for j in range(i+1, 9):
                dists.append(gdist[i][j])
                corrs.append(zz[i][j])
        dists, corrs = np.array(dists), np.array(corrs)

        from scipy.optimize import curve_fit
        def sphere_m(d, A, R): return A * np.cos(d/R)
        def flat_m(d, A, xi): return A * np.exp(-d/xi)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(dists, corrs, c='cyan', s=60, alpha=0.7, edgecolors='white', label='Data')

        d_fit = np.linspace(0.5, 6.5, 100)
        try:
            ps, _ = curve_fit(sphere_m, dists, corrs, p0=[0.1, 5])
            ax.plot(d_fit, sphere_m(d_fit, *ps), 'r-', lw=2, label=f'Sphere: cos(d/{ps[1]:.2f}), R={ps[1]:.2f}')
        except: pass
        try:
            pf, _ = curve_fit(flat_m, dists, corrs, p0=[0.1, 5])
            ax.plot(d_fit, flat_m(d_fit, *pf), 'g--', lw=2, label=f'Flat: exp(-d/{pf[1]:.2f})')
        except: pass

        ax.axhline(y=0, color='white', alpha=0.3)
        ax.set_xlabel('Graph Distance', fontsize=14, color='white')
        ax.set_ylabel('ZZ Correlation', fontsize=14, color='white')
        ax.set_title('Echo Geometry: Sphere vs Flat — The Curvature Test', fontsize=14, color='white')
        ax.legend(fontsize=11, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'curvature_test.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    curvature_test.png")

    # =====================================================================
    # PLOT 4: Mode Decomposition (3 bases)
    # =====================================================================
    if 'listen_echoes.json' in files:
        data = files['listen_echoes.json']

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(3)
        width = 0.25
        for bi, basis in enumerate(['Z', 'X', 'Y']):
            m = data[f'{basis}_modes']
            vals = [m['l0'], m['l1'], m['l2']]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1'][bi]
            ax.bar(x + bi*width, vals, width, label=f'{basis}-basis', color=colors, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(['$\\ell$=0\n(center)', '$\\ell$=1\n(ring-1)', '$\\ell$=2\n(ring-2)'],
                          fontsize=12, color='white')
        ax.set_ylabel('$\\Delta C^{(4)}$ Mode Amplitude', fontsize=14, color='white')
        ax.set_title('S$^2_3$ Spherical Harmonic Decomposition — Z, X, Y Bases\nFuzzy Sphere Modes on IBM Heron', fontsize=13, color='white')
        ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        ax.axhline(y=0, color='white', alpha=0.3)
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'mode_decomposition.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    mode_decomposition.png")

    # =====================================================================
    # PLOT 5: Butterfly-averaged scalar mode (if available)
    # =====================================================================
    if scalar_echo is not None and modes is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: per-qubit echo
        positions = {0:(0,0), 1:(1,0), 2:(2,0), 3:(3,0), 4:(4,0), 5:(5,0), 6:(6,0), 16:(3,1), 23:(3,2)}
        xs = [positions[pq][0] for pq in SPHERE_QUBITS]
        ys = [positions[pq][1] for pq in SPHERE_QUBITS]
        colors = [scalar_echo[PHYS_TO_LOG[pq]] for pq in SPHERE_QUBITS]

        sc = ax1.scatter(xs, ys, c=colors, cmap='RdBu_r', s=500, edgecolors='white', linewidth=2,
                        vmin=-max(abs(min(colors)), abs(max(colors))),
                        vmax=max(abs(min(colors)), abs(max(colors))))
        for pq in SPHERE_QUBITS:
            x, y = positions[pq]
            lq = PHYS_TO_LOG[pq]
            ring = "C" if pq == CENTER else ("1" if pq in RING1 else "2")
            ax1.annotate(f'q{pq}\n{scalar_echo[lq]:+.3f}', (x, y), ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        plt.colorbar(sc, ax=ax1, label='Butterfly-Averaged $\\Delta C^{(4)}$')
        ax1.set_title('Scalar Mode on S$^2_3$\n(Butterfly averaged over all 9 positions)', fontsize=12, color='white')
        ax1.set_facecolor('#1a1a2e')

        # Right: mode bar chart
        mode_vals = [modes['l0'], modes['l1'], modes['l2']]
        bars = ax2.bar(['$\\ell$=0', '$\\ell$=1', '$\\ell$=2'], mode_vals,
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1'], edgecolor='white')
        ax2.axhline(y=0, color='white', alpha=0.3)
        ax2.set_ylabel('Scalar Mode Amplitude', fontsize=12, color='white')
        ax2.set_title(f'Breathing Test: |$\\ell$=1|/|$\\ell$=0| = {abs(modes["l1"]/modes["l0"]):.4f}\n'
                     f'cos(1/$\\pi$) = {COS_BETA:.4f}', fontsize=12, color='white')
        ax2.set_facecolor('#1a1a2e')
        ax2.tick_params(colors='white')

        for ax in [ax1, ax2]:
            for spine in ax.spines.values(): spine.set_color('white')
        fig.patch.set_facecolor('#0d0d1a')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'scalar_mode_sphere.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    scalar_mode_sphere.png")

    # =====================================================================
    # PLOT 6: Sphere Depth Evolution
    # =====================================================================
    if 'sphere_s2n3.json' in files:
        data = files['sphere_s2n3.json']
        fig, ax = plt.subplots(figsize=(10, 6))
        depths = [r['depth'] for r in data['results']]
        l0s = [r['dc4_l0'] for r in data['results']]
        l1s = [r['dc4_l1'] for r in data['results']]
        l2s = [r['dc4_l2'] for r in data['results']]

        ax.plot(depths, l0s, 'ro-', markersize=10, label='$\\ell$=0 (center)', linewidth=2)
        ax.plot(depths, l1s, 'gs-', markersize=10, label='$\\ell$=1 (ring-1)', linewidth=2)
        ax.plot(depths, l2s, 'b^-', markersize=10, label='$\\ell$=2 (ring-2)', linewidth=2)
        ax.axhline(y=0, color='white', alpha=0.3)
        ax.set_xlabel('Circuit Depth', fontsize=14, color='white')
        ax.set_ylabel('Mode Amplitude', fontsize=14, color='white')
        ax.set_title('S$^2_3$ Mode Evolution with Scrambling Depth\n9 Qubits on IBM Heron — The Sphere Breathes', fontsize=13, color='white')
        ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#0d0d1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'sphere_depth_evolution.png'), dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    sphere_depth_evolution.png")

    print(f"  All plots saved to {FIGURES_DIR}")


def main():
    print("="*80)
    print("  FULL SEND — S^2_3 Butterfly Average + Visualizations")
    print("="*80)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=27)
    print(f"  Backend: {backend.name}")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend,
                                      initial_layout=SPHERE_QUBITS)

    # Run butterfly-averaged experiment
    all_dc4, scalar_echo, modes = run_butterfly_average(service, backend, pm, depth=6)

    # Save
    outfile = os.path.join(DATA_DIR, "butterfly_averaged.json")
    with open(outfile, 'w') as f:
        json.dump({
            'all_dc4': all_dc4.tolist(),
            'scalar_echo': scalar_echo.tolist(),
            'modes': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in modes.items()},
            'cos_beta': COS_BETA,
        }, f, indent=2)
    print(f"  Data saved: {outfile}")

    # Generate ALL plots
    make_plots(all_dc4, scalar_echo, modes)

    print(f"\n{'='*80}")
    print(f"  DONE.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
