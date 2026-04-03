"""
OTOC(2) Full Qubit Sweep — 10q to 156q on IBM Heron
====================================================
5000 shots per level, 5 MC instances + identity
Fixed M-B ratio = 0.75, depth capped for coherence
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
N_MC = 5
SHOTS = 5000
MB_RATIO = 0.75

# Sizes to sweep — all the way up to 156
SIZES = [10, 16, 20, 27, 36, 50, 70, 100, 130, 156]

# Depth scaling: enough to scramble but not exceed coherence
# Google used depth ~0.35-0.78 * N, decreasing ratio for larger N
# We cap depth to keep total CZ gates manageable
def get_depth(n):
    if n <= 20: return max(7, int(n * 0.7))
    elif n <= 50: return max(10, int(n * 0.5))
    else: return min(30, max(15, int(5 * math.sqrt(n))))


def build_otoc2(n_sys, depth, seed=42, d_shift=0,
                butterfly_qubit=None, measure_qubit=0,
                pauli_string1=None, pauli_string2=None):
    """Build OTOC(2) ABBA circuit."""
    if butterfly_qubit is None:
        butterfly_qubit = n_sys // 2

    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_sys)
    d = depth // 2 + d_shift

    def get_layers(start, end):
        layers = []
        rng_copy = np.random.RandomState(seed)
        # Advance RNG to the right position
        for _ in range(start):
            for q in range(n_sys):
                rng_copy.choice([0.25, 0.5, 0.75])
                rng_copy.uniform(-PI, PI)
        for layer in range(start, end):
            sq = []
            for q in range(n_sys):
                theta = rng_copy.choice([0.25, 0.5, 0.75]) * PI
                phi = rng_copy.uniform(-PI, PI)
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

    layers_first = get_layers(0, d)
    layers_second = get_layers(d, depth)

    # PASS 1
    apply_layers(qc, layers_first)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_second)
    qc.x(butterfly_qubit)
    apply_layers(qc, layers_second, inverse=True)
    apply_pauli(qc, pauli_string1)
    apply_layers(qc, layers_first, inverse=True)

    qc.z(measure_qubit)

    # PASS 2
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
    total = sum(counts.values())
    val = 0.0
    for bitstring, count in counts.items():
        bit = int(bitstring[-(qubit_idx+1)])
        val += (1 - 2*bit) * count
    return val / total


def main():
    print("="*75)
    print("  OTOC(2) FULL SWEEP — 10q to 156q")
    print("  IBM Heron, 5000 shots, cos(1/pi) breathing test")
    print("="*75)

    service = QiskitRuntimeService(token=IBM_TOKEN)
    backend = service.least_busy(min_num_qubits=156)
    print(f"  Backend: {backend.name} ({backend.num_qubits}q)")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

    all_results = []
    t_total_start = time.time()
    rng_mc = np.random.RandomState(7777)

    print(f"\n  {'N':>5s}  {'Depth':>5s}  {'B':>4s}  {'CZ':>6s}  {'TrDep':>6s}  "
          f"{'DC4(M)':>10s}  {'|DC4|avg':>10s}  {'QPU_s':>6s}  {'perQ':>8s}")
    print(f"  {'-----':>5s}  {'-----':>5s}  {'----':>4s}  {'------':>6s}  {'------':>6s}  "
          f"{'----------':>10s}  {'----------':>10s}  {'------':>6s}  {'--------':>8s}")

    prev_mean = None
    prev_n = None

    for n_sys in SIZES:
        depth = get_depth(n_sys)
        bf = int(n_sys * MB_RATIO)
        mq = 0

        # Build circuits
        circuits = []
        # Identity
        qc_id = build_otoc2(n_sys, depth, seed=42, butterfly_qubit=bf, measure_qubit=mq)
        circuits.append(qc_id)
        # MC randoms
        for mc in range(N_MC):
            s1 = rng_mc.choice(4, size=n_sys)
            s2 = rng_mc.choice(4, size=n_sys)
            qc_r = build_otoc2(n_sys, depth, seed=42, butterfly_qubit=bf,
                              measure_qubit=mq, pauli_string1=s1, pauli_string2=s2)
            circuits.append(qc_r)

        # Transpile
        try:
            transpiled = [pm.run(c) for c in circuits]
        except Exception as e:
            print(f"  {n_sys:5d}  TRANSPILE FAILED: {e}")
            continue

        ops = transpiled[0].count_ops()
        cz = ops.get('cz', 0)
        td = transpiled[0].depth()

        # Run on QPU
        try:
            t0 = time.time()
            sampler = SamplerV2(mode=backend)
            job = sampler.run(transpiled, shots=SHOTS)
            result = job.result()
            qpu_time = time.time() - t0
        except Exception as e:
            print(f"  {n_sys:5d}  QPU FAILED: {e}")
            continue

        # Extract results
        z_all = np.zeros((len(circuits), n_sys))
        for ci in range(len(circuits)):
            counts = result[ci].data.meas.get_counts()
            for qi in range(n_sys):
                z_all[ci, qi] = compute_z_exp(counts, n_sys, qi)

        z_id = z_all[0]
        z_rnd_mean = z_all[1:].mean(axis=0)
        dc4 = z_id - z_rnd_mean
        dc4_scalar = float(dc4[mq])
        dc4_mean = float(np.mean(np.abs(dc4)))

        # Per-qubit decay rate
        per_q_str = ""
        if prev_mean is not None and prev_mean > 1e-6 and dc4_mean > 1e-6:
            ratio = dc4_mean / prev_mean
            dn = n_sys - prev_n
            if dn > 0 and ratio > 0:
                per_q = ratio ** (1/dn)
                per_q_str = f"{per_q:.6f}"

        # Legendre decomposition
        from numpy.polynomial.legendre import legfit
        angles = np.linspace(0, PI, n_sys)
        x = np.cos(angles)
        try:
            leg = legfit(x, dc4, min(2, n_sys-1))
        except:
            leg = [0, 0, 0]

        print(f"  {n_sys:5d}  {depth:5d}  {bf:4d}  {cz:6d}  {td:6d}  "
              f"{dc4_scalar:+10.6f}  {dc4_mean:10.6f}  {qpu_time:6.1f}  {per_q_str:>8s}")

        all_results.append({
            'n_sys': n_sys, 'depth': depth, 'butterfly': bf,
            'cz_gates': cz, 'transpiled_depth': td,
            'qpu_time': qpu_time, 'job_id': job.job_id(),
            'dc4_scalar': dc4_scalar, 'dc4_mean_abs': dc4_mean,
            'dc4_all': dc4.tolist(), 'z_identity': z_id.tolist(),
            'z_random_mean': z_rnd_mean.tolist(),
            'legendre': leg.tolist() if hasattr(leg, 'tolist') else list(leg),
        })

        prev_mean = dc4_mean
        prev_n = n_sys

        elapsed = time.time() - t_total_start
        remaining = 510 - elapsed  # ~8.5 min budget
        if remaining < 30:
            print(f"\n  *** LOW TIME: {remaining:.0f}s remaining, stopping ***")
            break

    # =====================================================================
    # FINAL ANALYSIS
    # =====================================================================
    t_total = time.time() - t_total_start

    print(f"\n{'='*75}")
    print(f"  BREATHING ANALYSIS — cos(1/pi) = {COS_BETA:.6f}")
    print(f"{'='*75}")

    if len(all_results) >= 2:
        print(f"\n  Consecutive per-qubit decay rates:")
        for i in range(1, len(all_results)):
            r0 = all_results[i-1]
            r1 = all_results[i]
            m0 = r0['dc4_mean_abs']
            m1 = r1['dc4_mean_abs']
            dn = r1['n_sys'] - r0['n_sys']
            if m0 > 1e-6 and m1 > 1e-6 and dn > 0:
                ratio = m1 / m0
                per_q = ratio ** (1/dn)
                diff = abs(per_q - COS_BETA) / COS_BETA * 100
                match = "***MATCH***" if diff < 1.0 else ("~close~" if diff < 5 else "")
                print(f"    {r0['n_sys']:3d}q -> {r1['n_sys']:3d}q: "
                      f"ratio={ratio:.4f}, per_q={per_q:.6f}, "
                      f"diff={diff:.3f}% {match}")

        # Overall decay
        r_first = all_results[0]
        r_last = all_results[-1]
        m_first = r_first['dc4_mean_abs']
        m_last = r_last['dc4_mean_abs']
        dn_total = r_last['n_sys'] - r_first['n_sys']
        if m_first > 1e-6 and m_last > 1e-6 and dn_total > 0:
            ratio_total = m_last / m_first
            per_q_total = ratio_total ** (1/dn_total)
            print(f"\n    OVERALL {r_first['n_sys']}q -> {r_last['n_sys']}q: "
                  f"per_q={per_q_total:.6f}")
            print(f"    cos(1/pi) = {COS_BETA:.6f}")
            print(f"    DIFFERENCE: {abs(per_q_total-COS_BETA)/COS_BETA*100:.3f}%")

    # Legendre evolution
    print(f"\n  Legendre coefficient evolution:")
    print(f"  {'N':>5s}  {'l0':>10s}  {'l1':>10s}  {'l2':>10s}  {'|l1/l0|':>8s}  {'|l2/l1|':>8s}")
    for r in all_results:
        lc = r['legendre']
        if len(lc) >= 3:
            r10 = abs(lc[1]/lc[0]) if abs(lc[0]) > 1e-8 else 0
            r21 = abs(lc[2]/lc[1]) if abs(lc[1]) > 1e-8 else 0
            print(f"  {r['n_sys']:5d}  {lc[0]:+10.4f}  {lc[1]:+10.4f}  {lc[2]:+10.4f}  "
                  f"{r10:8.4f}  {r21:8.4f}")

    # Save
    outfile = os.path.join(DATA_DIR, "qpu_fullsweep.json")
    with open(outfile, 'w') as f:
        json.dump({
            'backend': backend.name,
            'total_wall_time': t_total,
            'shots': SHOTS, 'n_mc': N_MC,
            'cos_beta': COS_BETA,
            'results': all_results,
        }, f, indent=2)

    print(f"\n  Saved: {outfile}")
    print(f"  Total QPU time: {t_total:.0f}s")
    print(f"  Total circuits: {len(all_results) * (1 + N_MC)}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
