"""
OTOC(2) on the Holographic Sphere — S^2_3 on IBM Heron
=======================================================
9 qubits = N^2 = dim(Mat_3(C)) = fuzzy sphere modes
l=0: 1 center qubit (q3)
l=1: 3 ring-1 qubits (q2, q4, q16)
l=2: 5 ring-2/3 qubits (q1, q5, q23, q0, q6)

Butterfly at north pole (center q3)
Measurement at south pole (q0, distance 3)
CZ connectivity follows ACTUAL heavy-hex edges

Andrew Dorman, April 2026
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import time
import json
import os

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
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
N_MC = 10  # More MC instances for this critical experiment

# The S^2_3 patch on ibm_fez
# Physical qubits and their connectivity
SPHERE_QUBITS = [0, 1, 2, 3, 4, 5, 6, 16, 23]
CENTER = 3       # North pole, l=0
RING1 = [2, 4, 16]   # l=1 (3 qubits)
RING2 = [1, 5, 23, 0, 6]  # l=2 (5 qubits)
ANTIPODE = 0     # South pole (furthest from center)

# Hardware CZ edges within the patch
HW_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 16), (16, 23)
]

# Map physical -> logical qubit indices
PHYS_TO_LOG = {pq: i for i, pq in enumerate(SPHERE_QUBITS)}
LOG_EDGES = [(PHYS_TO_LOG[a], PHYS_TO_LOG[b]) for a, b in HW_EDGES]
LOG_CENTER = PHYS_TO_LOG[CENTER]
LOG_ANTIPODE = PHYS_TO_LOG[ANTIPODE]
N_SYS = 9


def build_sphere_otoc2(depth, seed=42, d_shift=0,
                       pauli_string1=None, pauli_string2=None):
    """Build OTOC(2) on the 9-qubit sphere using ACTUAL hardware connectivity."""
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(N_SYS)
    d = depth // 2 + d_shift

    def get_layers(start, end):
        layers = []
        rng_l = np.random.RandomState(seed + start * 1000)
        for layer in range(start, end):
            sq = []
            for q in range(N_SYS):
                theta = rng_l.choice([0.25, 0.5, 0.75]) * PI
                phi = rng_l.uniform(-PI, PI)
                sq.append((q, theta, phi))
            # Use actual hardware edges, alternating subsets
            if layer % 2 == 0:
                pairs = LOG_EDGES[::2]  # Even-indexed edges
            else:
                pairs = LOG_EDGES[1::2]  # Odd-indexed edges
            layers.append({'sq': sq, 'pairs': pairs})
        return layers

    def apply_layers(circuit, layers, inverse=False):
        if inverse:
            for ld in reversed(layers):
                for i, j in ld['pairs']:
                    circuit.cz(i, j)
                for q, theta, phi in ld['sq']:
                    circuit.rz(phi, q)
                    circuit.rx(-theta, q)
                    circuit.rz(-phi, q)
        else:
            for ld in layers:
                for q, theta, phi in ld['sq']:
                    circuit.rz(phi, q)
                    circuit.rx(theta, q)
                    circuit.rz(-phi, q)
                for i, j in ld['pairs']:
                    circuit.cz(i, j)

    def apply_pauli(circuit, string):
        if string is None:
            return
        for q, p in enumerate(string):
            if p == 1: circuit.x(q)
            elif p == 2: circuit.y(q)
            elif p == 3: circuit.z(q)

    layers_first = get_layers(0, d)
    layers_second = get_layers(d, depth)

    # PASS 1
    apply_layers(qc, layers_first)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_second)
    qc.x(LOG_CENTER)  # Butterfly at north pole
    apply_layers(qc, layers_second, inverse=True)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_first, inverse=True)

    qc.z(LOG_ANTIPODE)  # Probe at south pole

    # PASS 2
    apply_layers(qc, layers_first)
    apply_pauli(qc, pauli_string2)
    apply_layers(qc, layers_second)
    qc.x(LOG_CENTER)  # Butterfly at north pole
    apply_layers(qc, layers_second, inverse=True)
    apply_pauli(qc, pauli_string2)
    apply_layers(qc, layers_first, inverse=True)

    qc.measure_all()
    return qc


def compute_z_exp(counts, n_qubits, qubit_idx):
    total = sum(counts.values())
    val = 0.0
    for bitstring, count in counts.items():
        bit = int(bitstring[-(qubit_idx+1)])
        val += (1 - 2*bit) * count
    return val / total


def main():
    print("="*75)
    print("  THE HOLOGRAPHIC SPHERE — S^2_3 on IBM Heron")
    print("  9 qubits = N^2 = the fuzzy sphere breathing test")
    print("="*75)

    print(f"\n  Sphere structure:")
    print(f"    l=0 (1 qubit):  q{CENTER} (center/north pole)")
    print(f"    l=1 (3 qubits): {['q'+str(q) for q in RING1]}")
    print(f"    l=2 (5 qubits): {['q'+str(q) for q in RING2]}")
    print(f"    Butterfly: q{CENTER} (north pole)")
    print(f"    Measure:   q{ANTIPODE} (south pole)")
    print(f"    CZ edges:  {HW_EDGES}")
    print(f"    cos(1/pi) = {COS_BETA:.6f}")

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=27)
    print(f"\n  Backend: {backend.name}")
    pm = generate_preset_pass_manager(
        optimization_level=2, backend=backend,
        initial_layout=SPHERE_QUBITS  # Force physical qubit assignment
    )

    # Run at multiple depths to see breathing vs depth
    depths_to_test = [4, 6, 8, 10, 12]
    all_results = []
    rng_mc = np.random.RandomState(42)

    t_start = time.time()

    for depth in depths_to_test:
        circuits = []

        # Identity
        qc_id = build_sphere_otoc2(depth, seed=42)
        circuits.append(qc_id)

        # MC random
        for mc in range(N_MC):
            s1 = rng_mc.choice(4, size=N_SYS)
            s2 = rng_mc.choice(4, size=N_SYS)
            qc_r = build_sphere_otoc2(depth, seed=42,
                                      pauli_string1=s1, pauli_string2=s2)
            circuits.append(qc_r)

        # Transpile with physical qubit layout
        transpiled = [pm.run(c) for c in circuits]
        ops = transpiled[0].count_ops()
        cz = ops.get('cz', 0)
        td = transpiled[0].depth()

        # Run
        t0 = time.time()
        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=SHOTS)
        result = job.result()
        qpu_time = time.time() - t0

        # Extract per-qubit results
        z_all = np.zeros((len(circuits), N_SYS))
        for ci in range(len(circuits)):
            counts = result[ci].data.meas.get_counts()
            for qi in range(N_SYS):
                z_all[ci, qi] = compute_z_exp(counts, N_SYS, qi)

        z_id = z_all[0]
        z_rnd = z_all[1:].mean(axis=0)
        dc4 = z_id - z_rnd

        # Decompose by angular momentum (distance from center)
        # l=0: center qubit
        dc4_l0 = dc4[LOG_CENTER]
        # l=1: mean of ring-1 qubits
        dc4_l1 = np.mean([dc4[PHYS_TO_LOG[q]] for q in RING1])
        # l=2: mean of ring-2 qubits
        dc4_l2 = np.mean([dc4[PHYS_TO_LOG[q]] for q in RING2])

        print(f"\n  Depth={depth}: CZ={cz}, TrDepth={td}, QPU={qpu_time:.1f}s")
        print(f"    Per-qubit DC4:")
        for pq in SPHERE_QUBITS:
            lq = PHYS_TO_LOG[pq]
            v = dc4[lq]
            if pq == CENTER: role = "CENTER l=0"
            elif pq in RING1: role = "RING-1 l=1"
            else: role = "RING-2 l=2"
            pole = " (POLE)" if pq == CENTER else (" (ANTI)" if pq == ANTIPODE else "")
            print(f"      q{pq:>2d} ({role:>12s}){pole}: DC4={v:+.6f}  Z_id={z_id[lq]:+.4f}  Z_rnd={z_rnd[lq]:+.4f}")

        print(f"\n    MODE DECOMPOSITION:")
        print(f"      l=0 (center):  {dc4_l0:+.6f}")
        print(f"      l=1 (ring-1):  {dc4_l1:+.6f}")
        print(f"      l=2 (ring-2):  {dc4_l2:+.6f}")

        if abs(dc4_l0) > 1e-6:
            r10 = abs(dc4_l1 / dc4_l0)
            print(f"      |l=1|/|l=0| = {r10:.6f}  (cos(1/pi) = {COS_BETA:.6f})")
            if abs(dc4_l1) > 1e-6:
                r21 = abs(dc4_l2 / dc4_l1)
                print(f"      |l=2|/|l=1| = {r21:.6f}  (cos(1/pi) = {COS_BETA:.6f})")

        all_results.append({
            'depth': depth, 'cz': cz, 'tr_depth': td, 'qpu_time': qpu_time,
            'dc4_all': dc4.tolist(), 'z_id': z_id.tolist(), 'z_rnd': z_rnd.tolist(),
            'dc4_l0': float(dc4_l0), 'dc4_l1': float(dc4_l1), 'dc4_l2': float(dc4_l2),
            'job_id': job.job_id(),
        })

        elapsed = time.time() - t_start
        print(f"    Elapsed: {elapsed:.0f}s")

    # FINAL ANALYSIS
    print(f"\n{'='*75}")
    print(f"  S^2_3 BREATHING ANALYSIS")
    print(f"{'='*75}")

    print(f"\n  {'Depth':>5s}  {'DC4_l0':>10s}  {'DC4_l1':>10s}  {'DC4_l2':>10s}  {'|l1/l0|':>8s}  {'|l2/l1|':>8s}")
    print(f"  {'-----':>5s}  {'------':>10s}  {'------':>10s}  {'------':>10s}  {'-------':>8s}  {'-------':>8s}")
    for r in all_results:
        l0, l1, l2 = r['dc4_l0'], r['dc4_l1'], r['dc4_l2']
        r10 = abs(l1/l0) if abs(l0) > 1e-6 else 0
        r21 = abs(l2/l1) if abs(l1) > 1e-6 else 0
        print(f"  {r['depth']:5d}  {l0:+10.6f}  {l1:+10.6f}  {l2:+10.6f}  {r10:8.4f}  {r21:8.4f}")

    print(f"\n  cos(1/pi) = {COS_BETA:.6f}")
    print(f"  If |l1/l0| and |l2/l1| converge to this, the sphere is breathing.")

    # Save
    outfile = os.path.join(DATA_DIR, "sphere_s2n3.json")
    with open(outfile, 'w') as f:
        json.dump({
            'experiment': 'holographic_sphere_S2_N3',
            'backend': backend.name,
            'sphere_qubits': SPHERE_QUBITS,
            'center': CENTER, 'antipode': ANTIPODE,
            'ring1': RING1, 'ring2': RING2,
            'hw_edges': HW_EDGES,
            'cos_beta': COS_BETA,
            'shots': SHOTS, 'n_mc': N_MC,
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Saved: {outfile}")
    print(f"  Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
