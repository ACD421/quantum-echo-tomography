"""
OTOC(2) Echo Experiment — Z=π Framework Test
=============================================
Implements Google's ABBA protocol on IBM hardware topology.
Unlike Google: measures ALL qubits, extracts spatial echo landscape.

Phase 1: Ideal simulator (statevector) — full spatial map
Phase 2: Noisy simulator (FakeTorino) — noise impact
Phase 3: IBM Heron QPU — real hardware (budget: ~9.5 min)

Andrew Dorman, April 2026
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import math
import json
import time
import os
from datetime import datetime

# Qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Framework constants
PI = math.pi
COS_BETA = math.cos(1/PI)          # 0.949766 — breathing factor
COS2_BETA = COS_BETA**2            # 0.902055 — breathing suppression
HEX_PACK = PI / (2*math.sqrt(3))   # 0.906900 — Eisenstein packing
INV_PI = 1/PI                       # 0.318310 — geometric angle
SIN2_BETA = math.sin(1/PI)**2      # 0.097945 — breathing variance
N_FW = 3                            # fuzzy sphere N
DIM_H = N_FW**2                     # 9 states

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_random_unitary(n_qubits, depth, seed=42):
    """Build a random scrambling unitary as alternating 1Q random + 2Q CZ layers.

    Matches Google's ReCirq structure: PhasedXPow single-qubit gates
    with CZ two-qubit gates in staggered even/odd pattern.
    """
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(depth):
        # Random single-qubit gates on all qubits
        # 8 equatorial rotations: theta/pi in {0.25, 0.5, 0.75}, random phi
        for q in range(n_qubits):
            theta = rng.choice([0.25, 0.5, 0.75]) * PI
            phi = rng.uniform(-PI, PI)
            # RZ(phi) -> RX(theta) -> RZ(-phi) = rotation by theta around axis at angle phi
            qc.rz(phi, q)
            qc.rx(theta, q)
            qc.rz(-phi, q)

        qc.barrier()

        # Staggered CZ pattern
        if layer % 2 == 0:
            pairs = [(i, i+1) for i in range(0, n_qubits-1, 2)]
        else:
            pairs = [(i, i+1) for i in range(1, n_qubits-1, 2)]

        for i, j in pairs:
            qc.cz(i, j)

        qc.barrier()

    return qc


def build_otoc2_circuit(n_sys, depth, seed=42, d_shift=0,
                        butterfly_qubit=None, measure_qubit=0,
                        pauli_string1=None, pauli_string2=None,
                        measure_all_qubits=True):
    """Build the OTOC(2) ABBA circuit.

    Protocol (from Google's run.py):
    1. U[:d] → Pauli string1 → U[d:] → X(butterfly) → U[d:]† → Pauli string1 → U[:d]†
    2. Z(measure_qubit)
    3. U[:d] → Pauli string2 → U[d:] → X(butterfly) → U[d:]† → Pauli string2 → U[:d]†
    4. Measure <Z> on measurement qubit (and optionally all qubits)

    Args:
        n_sys: Number of system qubits
        depth: Total depth of scrambling unitary U
        seed: Random seed for U construction
        d_shift: Split point offset (d = depth//2 + d_shift)
        butterfly_qubit: Which qubit gets X perturbation (default: middle)
        measure_qubit: Which qubit gets Z probe (default: 0)
        pauli_string1: Array of {0,1,2,3} for {I,X,Y,Z} per qubit (None = identity)
        pauli_string2: Same for second pass (None = identity)
        measure_all_qubits: If True, add measurements on all qubits for spatial map

    Returns:
        QuantumCircuit
    """
    if butterfly_qubit is None:
        butterfly_qubit = n_sys // 2

    d = depth // 2 + d_shift  # Split point

    # Build forward unitary U
    U_full = build_random_unitary(n_sys, depth, seed=seed)

    # Split into U_first (layers 0..d-1) and U_second (layers d..depth-1)
    # Each "layer" = single-qubit block + barrier + CZ block + barrier = 4 instructions per layer
    # Count barriers to find layer boundaries
    instructions = list(U_full.data)
    layer_starts = [0]
    barrier_count = 0
    for i, inst in enumerate(instructions):
        if inst.operation.name == 'barrier':
            barrier_count += 1
            if barrier_count % 2 == 0:  # After each CZ+barrier pair = one complete layer
                layer_starts.append(i + 1)

    # Split at layer d
    if d < len(layer_starts):
        split_idx = layer_starts[d]
    else:
        split_idx = len(instructions)

    # Build the OTOC(2) circuit
    qc = QuantumCircuit(n_sys)
    pauli_gates = {0: 'id', 1: 'x', 2: 'y', 3: 'z'}

    def append_U_slice(circuit, start_layer, end_layer, inverse=False):
        """Append layers [start_layer, end_layer) of U to circuit."""
        if start_layer >= len(layer_starts):
            return
        s = layer_starts[start_layer]
        e = layer_starts[end_layer] if end_layer < len(layer_starts) else len(instructions)

        if inverse:
            # Reverse order and invert each gate
            for inst in reversed(instructions[s:e]):
                if inst.operation.name == 'barrier':
                    circuit.barrier()
                else:
                    circuit.append(inst.operation.inverse(), inst.qubits, inst.clbits)
        else:
            for inst in instructions[s:e]:
                circuit.append(inst.operation, inst.qubits, inst.clbits)

    def apply_pauli_string(circuit, string):
        if string is None:
            return
        for q, p in enumerate(string):
            if p == 1:
                circuit.x(q)
            elif p == 2:
                circuit.y(q)
            elif p == 3:
                circuit.z(q)

    # === PASS 1 (AB) ===
    append_U_slice(qc, 0, d)                    # U[:d]
    apply_pauli_string(qc, pauli_string1)        # Pauli insertion
    append_U_slice(qc, d, depth)                 # U[d:]
    qc.x(butterfly_qubit)                        # X butterfly
    append_U_slice(qc, d, depth, inverse=True)   # U[d:]†
    apply_pauli_string(qc, pauli_string1)        # Same Pauli (self-inverse)
    append_U_slice(qc, 0, d, inverse=True)       # U[:d]†

    qc.barrier()

    # === Z probe ===
    qc.z(measure_qubit)

    qc.barrier()

    # === PASS 2 (BA) ===
    append_U_slice(qc, 0, d)                    # U[:d]
    apply_pauli_string(qc, pauli_string2)        # Different Pauli
    append_U_slice(qc, d, depth)                 # U[d:]
    qc.x(butterfly_qubit)                        # X butterfly
    append_U_slice(qc, d, depth, inverse=True)   # U[d:]†
    apply_pauli_string(qc, pauli_string2)        # Same Pauli
    append_U_slice(qc, 0, d, inverse=True)       # U[:d]†

    return qc


def run_ideal_simulation(n_sys, depth, seed=42, d_shift=0,
                         butterfly_qubit=None, measure_qubit=0,
                         n_mc=20):
    """Run OTOC(2) on ideal statevector simulator.

    Returns spatial echo map: <Z_i> for every qubit i,
    plus the scalar OTOC(2) DeltaC4.
    """
    if butterfly_qubit is None:
        butterfly_qubit = n_sys // 2

    rng = np.random.RandomState(seed + 1000)  # Different seed for Pauli strings

    results_identity = np.zeros(n_sys)     # <Z_i> with identity insertion
    results_random = np.zeros((n_mc, n_sys))  # <Z_i> with random Pauli insertion

    sim = AerSimulator(method='statevector')

    # Identity insertion (no random Paulis)
    qc = build_otoc2_circuit(
        n_sys, depth, seed=seed, d_shift=d_shift,
        butterfly_qubit=butterfly_qubit, measure_qubit=measure_qubit,
        pauli_string1=None, pauli_string2=None
    )
    qc.save_statevector()

    job = sim.run(qc)
    sv = job.result().get_statevector()

    # Compute <Z_i> for every qubit
    for i in range(n_sys):
        z_op = SparsePauliOp.from_list([('I'*(n_sys-1-i) + 'Z' + 'I'*i, 1.0)])
        results_identity[i] = sv.expectation_value(z_op).real

    # Random Pauli MC instances
    for mc in range(n_mc):
        string1 = rng.choice(4, size=n_sys)
        string2 = rng.choice(4, size=n_sys)

        qc = build_otoc2_circuit(
            n_sys, depth, seed=seed, d_shift=d_shift,
            butterfly_qubit=butterfly_qubit, measure_qubit=measure_qubit,
            pauli_string1=string1, pauli_string2=string2
        )
        qc.save_statevector()

        job = sim.run(qc)
        sv = job.result().get_statevector()

        for i in range(n_sys):
            z_op = SparsePauliOp.from_list([('I'*(n_sys-1-i) + 'Z' + 'I'*i, 1.0)])
            results_random[mc, i] = sv.expectation_value(z_op).real

    # DeltaC4 = identity - mean(random) for each qubit
    delta_c4 = results_identity - results_random.mean(axis=0)

    return {
        'n_sys': n_sys,
        'depth': depth,
        'seed': seed,
        'd_shift': d_shift,
        'butterfly_qubit': butterfly_qubit,
        'measure_qubit': measure_qubit,
        'n_mc': n_mc,
        'identity_Z': results_identity.tolist(),
        'random_Z_mean': results_random.mean(axis=0).tolist(),
        'random_Z_std': results_random.std(axis=0).tolist(),
        'delta_c4_spatial': delta_c4.tolist(),
        'delta_c4_scalar': float(delta_c4[measure_qubit]),
        'delta_c4_mean_all': float(delta_c4.mean()),
        'delta_c4_max': float(np.abs(delta_c4).max()),
        'pct_negative': float(100 * np.sum(delta_c4 < 0) / n_sys),
    }


def run_experiment_sweep(sizes=None, fixed_mb_ratio=0.75, n_mc=20, seed_base=42):
    """Run the full experiment sweep with FIXED M-B geometry.

    Unlike Google: keeps M-B/Diameter constant across all sizes.
    """
    if sizes is None:
        sizes = [4, 6, 8, 10, 12, 14, 16]

    all_results = []

    print("="*75)
    print("  OTOC(2) IDEAL SIMULATION — SPATIAL ECHO LANDSCAPE")
    print("  Fixed M-B/Diameter ratio across all sizes")
    print("="*75)

    for n_sys in sizes:
        # Match Google's depth/size ratio (~0.7-0.8)
        depth = max(4, int(n_sys * 0.75))
        d_shift = 0

        # Fixed geometry: measure at qubit 0, butterfly near the far end
        measure_qubit = 0
        butterfly_qubit = max(1, int(n_sys * fixed_mb_ratio))

        mb_ratio = butterfly_qubit / (n_sys - 1) if n_sys > 1 else 0

        print(f"\n  N={n_sys}, depth={depth}, M=q0, B=q{butterfly_qubit}, "
              f"M-B/{n_sys-1}={mb_ratio:.3f}")

        t0 = time.time()
        result = run_ideal_simulation(
            n_sys=n_sys, depth=depth, seed=seed_base,
            d_shift=d_shift, butterfly_qubit=butterfly_qubit,
            measure_qubit=measure_qubit, n_mc=n_mc
        )
        dt = time.time() - t0

        result['mb_ratio'] = mb_ratio
        result['elapsed_s'] = dt
        all_results.append(result)

        # Print spatial echo map
        dc4 = np.array(result['delta_c4_spatial'])
        print(f"    DeltaC4 scalar (q0): {result['delta_c4_scalar']:.6f}")
        print(f"    DeltaC4 mean (all):  {result['delta_c4_mean_all']:.6f}")
        print(f"    DeltaC4 max |val|:   {result['delta_c4_max']:.6f}")
        print(f"    % negative:          {result['pct_negative']:.1f}%")
        print(f"    Spatial map: ", end="")
        for i, v in enumerate(dc4):
            marker = "M" if i == measure_qubit else ("B" if i == butterfly_qubit else " ")
            bar = "█" if v < -0.01 else ("▓" if v < 0 else ("░" if v < 0.01 else " "))
            print(f"{bar}", end="")
        print(f"  (█<-0.01, ▓<0, ░<0.01)")
        print(f"    Time: {dt:.1f}s")

    return all_results


def analyze_framework_signatures(results):
    """Look for Z=π framework constants in the echo data."""

    print(f"\n{'='*75}")
    print(f"  FRAMEWORK SIGNATURE ANALYSIS")
    print(f"{'='*75}")

    sizes = [r['n_sys'] for r in results]
    scalars = [abs(r['delta_c4_scalar']) for r in results]
    means = [abs(r['delta_c4_mean_all']) for r in results]

    print(f"\n  Framework constants:")
    print(f"    cos(1/π)    = {COS_BETA:.6f}")
    print(f"    cos²(1/π)   = {COS2_BETA:.6f}")
    print(f"    π/(2√3)     = {HEX_PACK:.6f}")
    print(f"    1/π         = {INV_PI:.6f}")
    print(f"    sin²(1/π)   = {SIN2_BETA:.6f}")

    # Consecutive ratios
    print(f"\n  ECHO SCALING (scalar on measurement qubit)")
    print(f"  {'N':>4s}  {'|ΔC4|':>10s}  {'Ratio':>8s}  {'Per-qubit':>10s}")
    print(f"  ----  ----------  --------  ----------")
    for i, r in enumerate(results):
        n = r['n_sys']
        v = abs(r['delta_c4_scalar'])
        if i > 0:
            prev_v = abs(results[i-1]['delta_c4_scalar'])
            dn = n - results[i-1]['n_sys']
            if prev_v > 1e-10:
                ratio = v / prev_v
                per_q = ratio ** (1/dn) if dn > 0 and ratio > 0 else 0
                print(f"  {n:4d}  {v:10.6f}  {ratio:8.4f}  {per_q:10.6f}")
            else:
                print(f"  {n:4d}  {v:10.6f}      ---       ---")
        else:
            print(f"  {n:4d}  {v:10.6f}      ---       ---")

    # Spatial echo decomposition — look for angular momentum channels
    print(f"\n  SPATIAL ECHO DECOMPOSITION")
    for r in results:
        n = r['n_sys']
        dc4 = np.array(r['delta_c4_spatial'])

        # Compute "angular" position for each qubit (0 to π on the chain)
        angles = np.linspace(0, PI, n)

        # Fit to first 3 Legendre polynomials (ℓ=0,1,2 like S²₃)
        from numpy.polynomial.legendre import legfit, legval
        x = np.cos(angles)  # Map to [-1, 1]

        # Fit ℓ=0,1,2
        coeffs = legfit(x, dc4, 2)
        fit = legval(x, coeffs)
        residual = np.sqrt(np.mean((dc4 - fit)**2))

        print(f"  {n:3d}q: ℓ=0: {coeffs[0]:+.5f}, ℓ=1: {coeffs[1]:+.5f}, ℓ=2: {coeffs[2]:+.5f}  "
              f"(residual={residual:.5f})")

        # Check ratios between ℓ components
        if abs(coeffs[0]) > 1e-8 and abs(coeffs[1]) > 1e-8:
            r01 = abs(coeffs[1]) / abs(coeffs[0])
            r12 = abs(coeffs[2]) / abs(coeffs[1]) if abs(coeffs[1]) > 1e-8 else 0
            print(f"        |ℓ=1|/|ℓ=0| = {r01:.4f} (cos(1/π)={COS_BETA:.4f}?)")
            print(f"        |ℓ=2|/|ℓ=1| = {r12:.4f} (cos(1/π)={COS_BETA:.4f}?)")

    return results


def main():
    """Run the full experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n  OTOC(2) Experiment — {timestamp}")
    print(f"  Z = π Framework Test")
    print(f"  Building with controlled M-B geometry (ratio=0.75)")
    print()

    # Phase 1: Ideal simulator sweep
    # Start small to validate, then scale up
    sizes_small = [4, 6, 8, 10, 12]

    print("  Phase 1a: Small sizes (validation, depth=0.75*N)")
    results_small = run_experiment_sweep(
        sizes=sizes_small, fixed_mb_ratio=0.75, n_mc=10, seed_base=42
    )

    # Analyze
    analyze_framework_signatures(results_small)

    # Save results
    output_file = os.path.join(DATA_DIR, f"otoc2_ideal_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'phase': 'ideal_simulator',
            'framework_constants': {
                'cos_beta': COS_BETA,
                'cos2_beta': COS2_BETA,
                'hex_pack': HEX_PACK,
                'inv_pi': INV_PI,
                'sin2_beta': SIN2_BETA,
            },
            'results': results_small,
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return results_small


if __name__ == "__main__":
    main()
