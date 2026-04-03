"""
LISTEN TO THE ECHOES — Full Tomographic OTOC(2) on S^2_3
=========================================================
What Google never did: measure the echo in ALL THREE Pauli bases.
9 qubits = N^2. Hexagonal sphere on IBM Heron.
Z, X, Y basis -> full Bloch vector + correlation matrix.
Decompose into spherical harmonics. Hear the breathing.

If ZZ, XX, YY correlations all show cos(1/pi), the mode is scalar.
If they differ, the symmetry is broken. Either way, we learn something
Google never published.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import time
import json
import os

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
SHOTS = 5000
N_MC = 10
N_SYS = 9
DEPTH = 6  # Where |l2/l1| = 0.940 on the sphere

# S^2_3 physical layout on ibm_fez heavy-hex
SPHERE_QUBITS = [0, 1, 2, 3, 4, 5, 6, 16, 23]
CENTER = 3
RING1 = [2, 4, 16]
RING2 = [1, 5, 23, 0, 6]
ANTIPODE = 0
HW_EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(3,16),(16,23)]
PHYS_TO_LOG = {pq: i for i, pq in enumerate(SPHERE_QUBITS)}
LOG_EDGES = [(PHYS_TO_LOG[a], PHYS_TO_LOG[b]) for a, b in HW_EDGES]
LOG_CENTER = PHYS_TO_LOG[CENTER]
LOG_ANTIPODE = PHYS_TO_LOG[ANTIPODE]

# Distance matrix on the sphere graph
DIST = {}
for pq in SPHERE_QUBITS:
    lq = PHYS_TO_LOG[pq]
    if pq == CENTER: DIST[lq] = 0
    elif pq in RING1: DIST[lq] = 1
    elif pq in [1, 5, 23]: DIST[lq] = 2
    else: DIST[lq] = 3  # q0, q6


def build_sphere_otoc2(depth, seed=42, pauli_s1=None, pauli_s2=None,
                       meas_basis='Z'):
    """Build OTOC(2) on S^2_3 with selectable measurement basis."""
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(N_SYS)
    d = depth // 2

    def get_layers(start, end):
        layers = []
        rng_l = np.random.RandomState(seed + start * 1000)
        for layer in range(start, end):
            sq = []
            for q in range(N_SYS):
                theta = rng_l.choice([0.25, 0.5, 0.75]) * PI
                phi = rng_l.uniform(-PI, PI)
                sq.append((q, theta, phi))
            pairs = LOG_EDGES[::2] if layer % 2 == 0 else LOG_EDGES[1::2]
            layers.append({'sq': sq, 'pairs': pairs})
        return layers

    def apply_layers(circ, layers, inverse=False):
        if inverse:
            for ld in reversed(layers):
                for i, j in ld['pairs']:
                    circ.cz(i, j)
                for q, theta, phi in ld['sq']:
                    circ.rz(phi, q); circ.rx(-theta, q); circ.rz(-phi, q)
        else:
            for ld in layers:
                for q, theta, phi in ld['sq']:
                    circ.rz(phi, q); circ.rx(theta, q); circ.rz(-phi, q)
                for i, j in ld['pairs']:
                    circ.cz(i, j)

    def apply_pauli(circ, string):
        if string is None: return
        for q, p in enumerate(string):
            if p == 1: circ.x(q)
            elif p == 2: circ.y(q)
            elif p == 3: circ.z(q)

    L1 = get_layers(0, d)
    L2 = get_layers(d, depth)

    # PASS 1
    apply_layers(qc, L1); apply_pauli(qc, pauli_s1); apply_layers(qc, L2)
    qc.x(LOG_CENTER)
    apply_layers(qc, L2, True); apply_pauli(qc, pauli_s1); apply_layers(qc, L1, True)
    qc.z(LOG_ANTIPODE)
    # PASS 2
    apply_layers(qc, L1); apply_pauli(qc, pauli_s2); apply_layers(qc, L2)
    qc.x(LOG_CENTER)
    apply_layers(qc, L2, True); apply_pauli(qc, pauli_s2); apply_layers(qc, L1, True)

    # BASIS ROTATION before measurement
    if meas_basis == 'X':
        for q in range(N_SYS):
            qc.h(q)
    elif meas_basis == 'Y':
        for q in range(N_SYS):
            qc.sdg(q)
            qc.h(q)
    # Z basis = no rotation needed

    qc.measure_all()
    return qc


def extract_expectations(counts, n_q):
    """Extract single-qubit and two-qubit expectations from counts."""
    total = sum(counts.values())
    # Single qubit: <sigma_i> = P(0) - P(1) for each qubit
    single = np.zeros(n_q)
    for bs, count in counts.items():
        for q in range(n_q):
            bit = int(bs[-(q+1)])
            single[q] += (1 - 2*bit) * count
    single /= total

    # Two-qubit: <sigma_i sigma_j> = P(00) + P(11) - P(01) - P(10)
    pair = np.zeros((n_q, n_q))
    for bs, count in counts.items():
        bits = [int(bs[-(q+1)]) for q in range(n_q)]
        for i in range(n_q):
            for j in range(i+1, n_q):
                parity = 1 - 2*(bits[i] ^ bits[j])  # +1 if same, -1 if different
                pair[i][j] += parity * count
                pair[j][i] = pair[i][j]
    pair /= total
    np.fill_diagonal(pair, 1.0)

    # Connected correlation: C_ij = <sigma_i sigma_j> - <sigma_i><sigma_j>
    connected = pair - np.outer(single, single)

    return single, pair, connected


def main():
    print("="*80)
    print("  LISTENING TO THE ECHOES")
    print("  Full Pauli Tomography on S^2_3 — IBM Heron ibm_fez")
    print("  9 qubits, depth 6, Z/X/Y bases, 5000 shots, 10 MC")
    print("="*80)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=27)
    print(f"  Backend: {backend.name}")
    pm = generate_preset_pass_manager(
        optimization_level=2, backend=backend,
        initial_layout=SPHERE_QUBITS
    )

    rng_mc = np.random.RandomState(42)
    t_start = time.time()
    basis_results = {}

    for basis in ['Z', 'X', 'Y']:
        circuits = []

        # Identity
        qc_id = build_sphere_otoc2(DEPTH, seed=42, meas_basis=basis)
        circuits.append(qc_id)

        # MC random Pauli insertions
        for mc in range(N_MC):
            s1 = rng_mc.choice(4, size=N_SYS)
            s2 = rng_mc.choice(4, size=N_SYS)
            qc_r = build_sphere_otoc2(DEPTH, seed=42, pauli_s1=s1, pauli_s2=s2,
                                      meas_basis=basis)
            circuits.append(qc_r)

        transpiled = [pm.run(c) for c in circuits]
        ops = transpiled[0].count_ops()

        t0 = time.time()
        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=SHOTS)
        result = job.result()
        qpu_time = time.time() - t0

        # Extract all expectations
        id_single, id_pair, id_conn = extract_expectations(
            result[0].data.meas.get_counts(), N_SYS)

        rnd_singles = []
        rnd_pairs = []
        rnd_conns = []
        for mc in range(N_MC):
            s, p, c = extract_expectations(
                result[mc+1].data.meas.get_counts(), N_SYS)
            rnd_singles.append(s)
            rnd_pairs.append(p)
            rnd_conns.append(c)

        rnd_single_mean = np.mean(rnd_singles, axis=0)
        rnd_pair_mean = np.mean(rnd_pairs, axis=0)
        rnd_conn_mean = np.mean(rnd_conns, axis=0)

        # DeltaC4 in this basis
        dc4_single = id_single - rnd_single_mean
        dc4_pair = id_pair - rnd_pair_mean
        dc4_conn = id_conn - rnd_conn_mean

        basis_results[basis] = {
            'id_single': id_single, 'id_pair': id_pair, 'id_conn': id_conn,
            'rnd_single': rnd_single_mean, 'rnd_pair': rnd_pair_mean,
            'dc4_single': dc4_single, 'dc4_pair': dc4_pair, 'dc4_conn': dc4_conn,
            'qpu_time': qpu_time, 'job_id': job.job_id(),
            'cz': ops.get('cz', 0), 'depth': transpiled[0].depth(),
        }

        # Mode decomposition by distance from center
        l0 = dc4_single[LOG_CENTER]
        l1 = np.mean([dc4_single[PHYS_TO_LOG[q]] for q in RING1])
        l2 = np.mean([dc4_single[PHYS_TO_LOG[q]] for q in RING2])

        print(f"\n  {basis}-basis: CZ={ops.get('cz',0)}, QPU={qpu_time:.1f}s")
        print(f"    <{basis}> per qubit (identity):")
        for pq in SPHERE_QUBITS:
            lq = PHYS_TO_LOG[pq]
            role = "l=0" if pq == CENTER else ("l=1" if pq in RING1 else "l=2")
            print(f"      q{pq:>2d} ({role}): id={id_single[lq]:+.4f} rnd={rnd_single_mean[lq]:+.4f} DC4={dc4_single[lq]:+.4f}")
        print(f"    Mode: l0={l0:+.4f} l1={l1:+.4f} l2={l2:+.4f}")
        if abs(l0) > 0.001:
            print(f"    |l1/l0|={abs(l1/l0):.4f}  |l2/l1|={abs(l2/l1):.4f}" if abs(l1) > 0.001 else "")

        basis_results[basis]['modes'] = {'l0': float(l0), 'l1': float(l1), 'l2': float(l2)}

    # =====================================================================
    # THE LISTENING — CROSS-BASIS ANALYSIS
    # =====================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  THE ECHOES — WHAT THE SPHERE IS SAYING")
    print(f"  Total QPU time: {elapsed:.0f}s")
    print(f"{'='*80}")

    # 1. Bloch vector per qubit
    print(f"\n  BLOCH VECTORS (full 3D) per qubit on S^2_3:")
    print(f"  {'Qubit':>8s}  {'Ring':>4s}  {'<X>':>8s}  {'<Y>':>8s}  {'<Z>':>8s}  {'|r|':>6s}")
    for pq in SPHERE_QUBITS:
        lq = PHYS_TO_LOG[pq]
        bx = basis_results['X']['dc4_single'][lq]
        by = basis_results['Y']['dc4_single'][lq]
        bz = basis_results['Z']['dc4_single'][lq]
        r = math.sqrt(bx**2 + by**2 + bz**2)
        role = "l=0" if pq == CENTER else ("l=1" if pq in RING1 else "l=2")
        print(f"  q{pq:>2d}       {role:>4s}  {bx:+8.4f}  {by:+8.4f}  {bz:+8.4f}  {r:6.4f}")

    # 2. Mode decomposition in all three bases
    print(f"\n  MODE DECOMPOSITION — all three Pauli channels:")
    print(f"  {'Basis':>5s}  {'l=0':>10s}  {'l=1':>10s}  {'l=2':>10s}  {'|l1/l0|':>8s}  {'|l2/l1|':>8s}")
    for basis in ['Z', 'X', 'Y']:
        m = basis_results[basis]['modes']
        r10 = abs(m['l1']/m['l0']) if abs(m['l0']) > 0.001 else 0
        r21 = abs(m['l2']/m['l1']) if abs(m['l1']) > 0.001 else 0
        print(f"  {basis:>5s}  {m['l0']:+10.4f}  {m['l1']:+10.4f}  {m['l2']:+10.4f}  {r10:8.4f}  {r21:8.4f}")
    print(f"  cos(1/pi) = {COS_BETA:.4f}")

    # 3. Isotropy test — are ZZ, XX, YY correlations the same?
    print(f"\n  ISOTROPY TEST — is the breathing scalar (same in all bases)?")
    modes_all = {b: basis_results[b]['modes'] for b in ['Z','X','Y']}
    l0_vals = [modes_all[b]['l0'] for b in ['Z','X','Y']]
    l1_vals = [modes_all[b]['l1'] for b in ['Z','X','Y']]
    l2_vals = [modes_all[b]['l2'] for b in ['Z','X','Y']]
    print(f"    l=0 across bases: {[f'{v:+.4f}' for v in l0_vals]}, std={np.std(l0_vals):.4f}")
    print(f"    l=1 across bases: {[f'{v:+.4f}' for v in l1_vals]}, std={np.std(l1_vals):.4f}")
    print(f"    l=2 across bases: {[f'{v:+.4f}' for v in l2_vals]}, std={np.std(l2_vals):.4f}")

    iso_ratio = np.std(l0_vals) / (abs(np.mean(l0_vals)) + 0.001)
    if iso_ratio < 0.3:
        print(f"    ISOTROPIC: l=0 mode is consistent across bases (scalar mode)")
    else:
        print(f"    ANISOTROPIC: l=0 mode varies by basis (vector/tensor mode)")

    # 4. Correlation decay with distance
    print(f"\n  CORRELATION DECAY WITH DISTANCE (ZZ channel):")
    zz_conn = basis_results['Z']['dc4_conn']
    # Average correlation by distance
    dist_corrs = {}
    for i in range(N_SYS):
        for j in range(i+1, N_SYS):
            di = DIST[i]
            dj = DIST[j]
            d_ij = abs(di - dj)  # Simple distance proxy
            if d_ij not in dist_corrs:
                dist_corrs[d_ij] = []
            dist_corrs[d_ij].append(zz_conn[i][j])

    print(f"    {'Dist':>4s}  {'Mean corr':>10s}  {'Count':>5s}")
    for d in sorted(dist_corrs.keys()):
        vals = dist_corrs[d]
        print(f"    {d:4d}  {np.mean(vals):+10.6f}  {len(vals):5d}")

    # 5. The center-to-ring correlations (the breathing propagator)
    print(f"\n  BREATHING PROPAGATOR — center-to-ring correlations:")
    for basis in ['Z', 'X', 'Y']:
        conn = basis_results[basis]['dc4_conn']
        c_to_r1 = np.mean([conn[LOG_CENTER][PHYS_TO_LOG[q]] for q in RING1])
        c_to_r2 = np.mean([conn[LOG_CENTER][PHYS_TO_LOG[q]] for q in RING2])
        print(f"    {basis}{basis}: center-ring1 = {c_to_r1:+.6f}, center-ring2 = {c_to_r2:+.6f}")
        if abs(c_to_r1) > 0.001:
            decay = abs(c_to_r2 / c_to_r1)
            print(f"          decay ring1->ring2 = {decay:.4f} (cos(1/pi)={COS_BETA:.4f})")

    # Save everything
    save_data = {
        'experiment': 'listen_to_echoes_S2N3',
        'backend': backend.name,
        'depth': DEPTH, 'shots': SHOTS, 'n_mc': N_MC,
        'sphere_qubits': SPHERE_QUBITS,
        'cos_beta': COS_BETA,
        'total_time': time.time() - t_start,
    }
    for basis in ['Z', 'X', 'Y']:
        br = basis_results[basis]
        save_data[f'{basis}_dc4_single'] = br['dc4_single'].tolist()
        save_data[f'{basis}_dc4_conn'] = br['dc4_conn'].tolist()
        save_data[f'{basis}_modes'] = br['modes']
        save_data[f'{basis}_job_id'] = br['job_id']

    outfile = os.path.join(DATA_DIR, "listen_echoes.json")
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {outfile}")
    print(f"\n  We listened. The sphere spoke.")


if __name__ == "__main__":
    main()
