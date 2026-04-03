"""
OTOC(2) Full QPU Experiment — Options A + B
============================================
A: Size sweep (10, 16, 20 qubits) with fixed M-B ratio
B: Geometry test (12 qubits, 3 butterfly distances)

5 MC instances + 1 identity per configuration = 6 circuits each
Total: 6 configs x 6 circuits = 36 circuits @ ~4s = ~144s QPU time

Andrew Dorman, April 2026
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
COS2_BETA = COS_BETA**2
HEX_PACK = PI / (2*math.sqrt(3))
N_MC = 5  # Random Pauli instances per configuration
SHOTS = 2048


def build_otoc2(n_sys, depth, seed=42, d_shift=0,
                butterfly_qubit=None, measure_qubit=0,
                pauli_string1=None, pauli_string2=None):
    """Build OTOC(2) ABBA circuit with measure_all."""
    if butterfly_qubit is None:
        butterfly_qubit = n_sys // 2

    rng_u = np.random.RandomState(seed)
    qc = QuantumCircuit(n_sys)
    d = depth // 2 + d_shift

    def get_layers(rng, start, end):
        """Generate layer data for layers [start, end)."""
        layers = []
        # Reset RNG to produce consistent layers
        for layer in range(start, end):
            sq = []
            for q in range(n_sys):
                theta = rng.choice([0.25, 0.5, 0.75]) * PI
                phi = rng.uniform(-PI, PI)
                sq.append((q, theta, phi))
            if layer % 2 == 0:
                pairs = [(i, i+1) for i in range(0, n_sys-1, 2)]
            else:
                pairs = [(i, i+1) for i in range(1, n_sys-1, 2)]
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

    # Generate all layers deterministically
    rng_u_copy = np.random.RandomState(seed)
    layers_first = get_layers(rng_u_copy, 0, d)
    layers_second = get_layers(rng_u_copy, d, depth)

    # === PASS 1 ===
    apply_layers(qc, layers_first)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_second)
    qc.x(butterfly_qubit)
    apply_layers(qc, layers_second, inverse=True)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_first, inverse=True)

    # Z probe
    qc.z(measure_qubit)

    # === PASS 2 ===
    apply_layers(qc, layers_first)
    apply_pauli(qc, pauli_string2)
    apply_layers(qc, layers_second)
    qc.x(butterfly_qubit)
    apply_layers(qc, layers_second, inverse=True)
    apply_pauli(qc, pauli_string2)
    apply_layers(qc, layers_first, inverse=True)

    qc.measure_all()
    return qc


def compute_z_exp(counts, n_qubits, qubit_idx):
    """<Z_i> from shot counts."""
    total = sum(counts.values())
    val = 0.0
    for bitstring, count in counts.items():
        bit = int(bitstring[-(qubit_idx+1)])
        val += (1 - 2*bit) * count
    return val / total


def run_config(service, backend, pm, n_sys, depth, butterfly_qubit,
               measure_qubit, config_name, seed=42):
    """Run one OTOC(2) configuration: identity + N_MC random insertions."""

    rng = np.random.RandomState(seed + 7777)
    circuits = []
    labels = []

    # Identity circuit
    qc_id = build_otoc2(n_sys, depth, seed=seed, butterfly_qubit=butterfly_qubit,
                        measure_qubit=measure_qubit)
    circuits.append(qc_id)
    labels.append('identity')

    # MC random Pauli circuits
    for mc in range(N_MC):
        s1 = rng.choice(4, size=n_sys)
        s2 = rng.choice(4, size=n_sys)
        qc_rnd = build_otoc2(n_sys, depth, seed=seed, butterfly_qubit=butterfly_qubit,
                             measure_qubit=measure_qubit,
                             pauli_string1=s1, pauli_string2=s2)
        circuits.append(qc_rnd)
        labels.append(f'random_{mc}')

    # Transpile all
    t0 = time.time()
    transpiled = [pm.run(c) for c in circuits]
    t_transpile = time.time() - t0

    ops = transpiled[0].count_ops()
    cz_count = ops.get('cz', 0)
    t_depth = transpiled[0].depth()

    print(f"\n  [{config_name}] {n_sys}q, depth={depth}, M=q{measure_qubit}, B=q{butterfly_qubit}")
    print(f"    Transpiled: depth={t_depth}, CZ={cz_count}, transpile_time={t_transpile:.1f}s")

    # Submit to QPU
    t_qpu_start = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    print(f"    Job: {job.job_id()} — waiting...")
    result = job.result()
    t_qpu = time.time() - t_qpu_start
    print(f"    QPU wall time: {t_qpu:.1f}s")

    # Extract all results
    z_all = np.zeros((len(circuits), n_sys))
    for ci in range(len(circuits)):
        counts = result[ci].data.meas.get_counts()
        for qi in range(n_sys):
            z_all[ci, qi] = compute_z_exp(counts, n_sys, qi)

    z_identity = z_all[0]
    z_random_mean = z_all[1:].mean(axis=0)
    z_random_std = z_all[1:].std(axis=0)
    delta_c4 = z_identity - z_random_mean

    # Legendre decomposition
    from numpy.polynomial.legendre import legfit
    angles = np.linspace(0, PI, n_sys)
    x = np.cos(angles)
    leg_coeffs = legfit(x, delta_c4, min(2, n_sys-1))

    # Print spatial map
    print(f"    Spatial DeltaC4:")
    print(f"    {'Q':>4s}  {'Z_id':>8s}  {'Z_rnd':>8s}  {'DC4':>8s}  {'':>3s}")
    for i in range(n_sys):
        m = "M" if i == measure_qubit else ("B" if i == butterfly_qubit else " ")
        bar = "███" if delta_c4[i] < -0.01 else ("▓▓▓" if delta_c4[i] < 0 else ("░░░" if delta_c4[i] < 0.05 else "   "))
        print(f"    q{i:>2d}{m} {z_identity[i]:8.4f}  {z_random_mean[i]:8.4f}  {delta_c4[i]:+8.4f}  {bar}")

    print(f"    Scalar DC4 (q{measure_qubit}): {delta_c4[measure_qubit]:+.6f}")
    print(f"    Mean |DC4|:              {np.mean(np.abs(delta_c4)):.6f}")
    print(f"    % negative:              {100*np.sum(delta_c4<0)/n_sys:.1f}%")

    if len(leg_coeffs) >= 3:
        print(f"    Legendre: l0={leg_coeffs[0]:+.4f}, l1={leg_coeffs[1]:+.4f}, l2={leg_coeffs[2]:+.4f}")
        if abs(leg_coeffs[0]) > 1e-6:
            print(f"    |l1/l0|={abs(leg_coeffs[1]/leg_coeffs[0]):.4f}, |l2/l1|={abs(leg_coeffs[2]/leg_coeffs[1]):.4f}" if abs(leg_coeffs[1]) > 1e-6 else "")

    return {
        'config_name': config_name,
        'n_sys': n_sys,
        'depth': depth,
        'butterfly_qubit': butterfly_qubit,
        'measure_qubit': measure_qubit,
        'mb_distance': butterfly_qubit - measure_qubit,
        'mb_ratio': (butterfly_qubit - measure_qubit) / (n_sys - 1),
        'transpiled_depth': t_depth,
        'transpiled_cz': cz_count,
        'qpu_wall_time': t_qpu,
        'shots': SHOTS,
        'n_mc': N_MC,
        'job_id': job.job_id(),
        'z_identity': z_identity.tolist(),
        'z_random_mean': z_random_mean.tolist(),
        'z_random_std': z_random_std.tolist(),
        'delta_c4': delta_c4.tolist(),
        'delta_c4_scalar': float(delta_c4[measure_qubit]),
        'legendre_coeffs': leg_coeffs.tolist(),
    }


def main():
    print("="*75)
    print("  OTOC(2) FULL EXPERIMENT — OPTIONS A + B")
    print("  IBM Heron QPU, Z=pi Framework Test")
    print("="*75)

    # Connect
    print("\n  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=21)
    print(f"  Backend: {backend.name} ({backend.num_qubits}q)")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

    all_results = []
    t_total_start = time.time()

    # =====================================================================
    # OPTION A: SIZE SWEEP (fixed M-B ratio = 0.75)
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"  OPTION A: SIZE SWEEP (M-B/Diam = 0.75)")
    print(f"{'='*75}")

    for n_sys, depth in [(10, 7), (16, 12), (20, 15)]:
        bf = int(n_sys * 0.75)
        result = run_config(service, backend, pm,
                           n_sys=n_sys, depth=depth,
                           butterfly_qubit=bf, measure_qubit=0,
                           config_name=f"A_{n_sys}q", seed=42)
        all_results.append(result)

        elapsed = time.time() - t_total_start
        print(f"    Total elapsed: {elapsed:.0f}s")

    # =====================================================================
    # OPTION B: GEOMETRY TEST (12 qubits, 3 butterfly distances)
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"  OPTION B: GEOMETRY TEST (12q, varying M-B distance)")
    print(f"{'='*75}")

    for bf, label in [(3, "close"), (6, "medium"), (9, "far")]:
        mb_ratio = bf / 11
        result = run_config(service, backend, pm,
                           n_sys=12, depth=9,
                           butterfly_qubit=bf, measure_qubit=0,
                           config_name=f"B_12q_bf{bf}_{label}", seed=42)
        all_results.append(result)

        elapsed = time.time() - t_total_start
        print(f"    Total elapsed: {elapsed:.0f}s")

    # =====================================================================
    # ANALYSIS
    # =====================================================================
    t_total = time.time() - t_total_start

    print(f"\n{'='*75}")
    print(f"  EXPERIMENT COMPLETE — ANALYSIS")
    print(f"{'='*75}")

    print(f"\n  Total QPU wall time: {t_total:.0f}s")
    print(f"  Total circuits: {len(all_results) * (1 + N_MC)}")

    # Option A analysis
    print(f"\n  --- OPTION A: SIZE SCALING ---")
    print(f"  {'Config':>12s}  {'N':>4s}  {'DC4(M)':>10s}  {'|DC4|mean':>10s}  {'CZ':>5s}  {'Depth':>6s}")
    print(f"  {'--------':>12s}  ----  ----------  ----------  -----  ------")
    a_results = [r for r in all_results if r['config_name'].startswith('A_')]
    for r in a_results:
        print(f"  {r['config_name']:>12s}  {r['n_sys']:4d}  {r['delta_c4_scalar']:+10.6f}  "
              f"{np.mean(np.abs(r['delta_c4'])):10.6f}  {r['transpiled_cz']:5d}  {r['transpiled_depth']:6d}")

    # Size scaling ratios
    if len(a_results) >= 2:
        print(f"\n  Size scaling ratios:")
        for i in range(1, len(a_results)):
            r0, r1 = a_results[i-1], a_results[i]
            v0 = abs(r0['delta_c4_scalar'])
            v1 = abs(r1['delta_c4_scalar'])
            dn = r1['n_sys'] - r0['n_sys']
            if v0 > 1e-8:
                ratio = v1 / v0
                per_q = ratio ** (1/dn) if ratio > 0 else 0
                print(f"    {r0['n_sys']}q -> {r1['n_sys']}q: ratio={ratio:.4f}, per_qubit={per_q:.4f}")
                print(f"      Compare: cos(1/pi)={COS_BETA:.4f}, 1/pi={1/PI:.4f}")

    # Option B analysis
    print(f"\n  --- OPTION B: GEOMETRY (M-B DISTANCE) ---")
    print(f"  {'Config':>20s}  {'M-B':>5s}  {'Ratio':>6s}  {'DC4(M)':>10s}  {'|DC4|mean':>10s}")
    print(f"  {'--------':>20s}  -----  ------  ----------  ----------")
    b_results = [r for r in all_results if r['config_name'].startswith('B_')]
    for r in b_results:
        print(f"  {r['config_name']:>20s}  {r['mb_distance']:5d}  {r['mb_ratio']:6.3f}  "
              f"{r['delta_c4_scalar']:+10.6f}  {np.mean(np.abs(r['delta_c4'])):10.6f}")

    # Legendre comparison
    print(f"\n  --- LEGENDRE DECOMPOSITION (all configs) ---")
    print(f"  {'Config':>20s}  {'l0':>8s}  {'l1':>8s}  {'l2':>8s}  {'|l1/l0|':>8s}  {'|l2/l1|':>8s}")
    print(f"  {'--------':>20s}  --------  --------  --------  --------  --------")
    for r in all_results:
        lc = r['legendre_coeffs']
        if len(lc) >= 3:
            r10 = abs(lc[1]/lc[0]) if abs(lc[0]) > 1e-8 else 0
            r21 = abs(lc[2]/lc[1]) if abs(lc[1]) > 1e-8 else 0
            print(f"  {r['config_name']:>20s}  {lc[0]:+8.4f}  {lc[1]:+8.4f}  {lc[2]:+8.4f}  "
                  f"{r10:8.4f}  {r21:8.4f}")

    print(f"\n  Framework constants for comparison:")
    print(f"    cos(1/pi)   = {COS_BETA:.6f}")
    print(f"    cos^2(1/pi) = {COS2_BETA:.6f}")
    print(f"    pi/(2sqrt3) = {HEX_PACK:.6f}")
    print(f"    1/pi        = {1/PI:.6f}")
    print(f"    2/pi        = {2/PI:.6f}")

    # Save everything
    outfile = os.path.join(DATA_DIR, "qpu_full_experiment.json")
    with open(outfile, 'w') as f:
        json.dump({
            'backend': backend.name,
            'total_wall_time_s': t_total,
            'shots': SHOTS,
            'n_mc': N_MC,
            'framework_constants': {
                'cos_beta': COS_BETA, 'cos2_beta': COS2_BETA,
                'hex_pack': HEX_PACK, 'inv_pi': 1/PI,
            },
            'results': all_results,
        }, f, indent=2)
    print(f"\n  All results saved to: {outfile}")

    print(f"\n{'='*75}")
    print(f"  DONE. Total wall time: {t_total:.0f}s")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
