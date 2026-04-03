"""
OTOC(2) QPU Baseline Test — 10 qubits, minimal shots
=====================================================
Goal: measure how much QPU time one circuit costs.
Then pause for budget decision.
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


def build_random_unitary(n_qubits, depth, seed=42):
    """Random scrambling unitary: alternating 1Q random + CZ layers."""
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)
    PI = math.pi

    for layer in range(depth):
        for q in range(n_qubits):
            theta = rng.choice([0.25, 0.5, 0.75]) * PI
            phi = rng.uniform(-PI, PI)
            qc.rz(phi, q)
            qc.rx(theta, q)
            qc.rz(-phi, q)

        if layer % 2 == 0:
            pairs = [(i, i+1) for i in range(0, n_qubits-1, 2)]
        else:
            pairs = [(i, i+1) for i in range(1, n_qubits-1, 2)]
        for i, j in pairs:
            qc.cz(i, j)

    return qc


def build_otoc2_hardware(n_sys, depth, seed=42, d_shift=0,
                         butterfly_qubit=None, measure_qubit=0,
                         pauli_string1=None, pauli_string2=None):
    """Build OTOC(2) ABBA circuit for hardware (with measurements)."""
    if butterfly_qubit is None:
        butterfly_qubit = n_sys // 2

    PI = math.pi
    rng_u = np.random.RandomState(seed)

    # Build U as a list of gate instructions
    # We'll construct the full ABBA manually for clarity
    qc = QuantumCircuit(n_sys)

    d = depth // 2 + d_shift  # Split point

    def add_scramble_layers(circuit, start_layer, end_layer, rng, inverse=False):
        """Add layers [start, end) of the scrambling unitary."""
        layers_data = []
        for layer in range(start_layer, end_layer):
            layer_info = {'sq': [], 'pairs': []}
            for q in range(n_sys):
                theta = rng.choice([0.25, 0.5, 0.75]) * PI
                phi = rng.uniform(-PI, PI)
                layer_info['sq'].append((q, theta, phi))

            if layer % 2 == 0:
                pairs = [(i, i+1) for i in range(0, n_sys-1, 2)]
            else:
                pairs = [(i, i+1) for i in range(1, n_sys-1, 2)]
            layer_info['pairs'] = pairs
            layers_data.append(layer_info)

        if inverse:
            layers_data = layers_data[::-1]

        for ld in layers_data:
            if inverse:
                # CZ first (self-inverse), then inverse single-qubit
                for i, j in ld['pairs']:
                    circuit.cz(i, j)
                for q, theta, phi in ld['sq']:
                    circuit.rz(phi, q)
                    circuit.rx(-theta, q)
                    circuit.rz(-phi, q)
            else:
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

    # We need deterministic RNG states for U and U_inverse
    # Save RNG state before each segment
    rng_state_0 = rng_u.get_state()

    # === PASS 1 ===
    rng_u.set_state(rng_state_0)
    add_scramble_layers(qc, 0, d, rng_u)             # U[:d]
    rng_state_d = rng_u.get_state()
    apply_pauli(qc, pauli_string1)                     # Pauli 1
    rng_u.set_state(rng_state_d)
    add_scramble_layers(qc, d, depth, rng_u)           # U[d:]
    qc.x(butterfly_qubit)                               # X butterfly
    rng_u.set_state(rng_state_d)
    add_scramble_layers(qc, d, depth, rng_u, inverse=True)  # U[d:]†
    apply_pauli(qc, pauli_string1)                     # Pauli 1 (self-inverse)
    rng_u.set_state(rng_state_0)
    add_scramble_layers(qc, 0, d, rng_u, inverse=True)  # U[:d]†

    # === Z probe ===
    qc.z(measure_qubit)

    # === PASS 2 ===
    rng_u.set_state(rng_state_0)
    add_scramble_layers(qc, 0, d, rng_u)             # U[:d]
    rng_state_d2 = rng_u.get_state()
    apply_pauli(qc, pauli_string2)                     # Pauli 2
    rng_u.set_state(rng_state_d2)
    add_scramble_layers(qc, d, depth, rng_u)           # U[d:]
    qc.x(butterfly_qubit)                               # X butterfly
    rng_u.set_state(rng_state_d2)
    add_scramble_layers(qc, d, depth, rng_u, inverse=True)  # U[d:]†
    apply_pauli(qc, pauli_string2)                     # Pauli 2
    rng_u.set_state(rng_state_0)
    add_scramble_layers(qc, 0, d, rng_u, inverse=True)  # U[:d]†

    # Measure all qubits
    qc.measure_all()

    return qc


def compute_z_expectation(counts, n_qubits, qubit_idx):
    """Compute <Z_i> from measurement counts."""
    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        # Qiskit bitstrings are reversed: bit 0 is rightmost
        bit = int(bitstring[-(qubit_idx+1)])
        exp_val += (1 - 2*bit) * count  # |0> -> +1, |1> -> -1
    return exp_val / total


def main():
    print("="*75)
    print("  OTOC(2) QPU BASELINE TEST")
    print("  10 qubits, 1024 shots, identity insertion")
    print("="*75)

    # Connect to IBM
    print("\n  Connecting to IBM Quantum...")
    t0 = time.time()
    service = QiskitRuntimeService(token=IBM_TOKEN)
    print(f"  Connected in {time.time()-t0:.1f}s")

    # Get least busy backend with enough qubits
    print("  Finding least busy backend (21+ qubits)...")
    backend = service.least_busy(min_num_qubits=21)
    print(f"  Backend: {backend.name}")
    print(f"  Qubits: {backend.num_qubits}")

    # Build the circuit
    n_sys = 10
    depth = 7  # ~0.7 * n_sys
    butterfly_qubit = 7  # ~75% of chain
    measure_qubit = 0

    print(f"\n  Building OTOC(2) circuit:")
    print(f"    N={n_sys}, depth={depth}, M=q{measure_qubit}, B=q{butterfly_qubit}")

    # Identity insertion (no random Paulis)
    qc_identity = build_otoc2_hardware(
        n_sys, depth, seed=42, d_shift=0,
        butterfly_qubit=butterfly_qubit, measure_qubit=measure_qubit,
        pauli_string1=None, pauli_string2=None
    )
    print(f"  Raw circuit: {qc_identity.num_qubits}q, depth={qc_identity.depth()}, "
          f"gates={sum(qc_identity.count_ops().values())}")

    # One random Pauli insertion for comparison
    rng = np.random.RandomState(123)
    s1 = rng.choice(4, size=n_sys)
    s2 = rng.choice(4, size=n_sys)
    qc_random = build_otoc2_hardware(
        n_sys, depth, seed=42, d_shift=0,
        butterfly_qubit=butterfly_qubit, measure_qubit=measure_qubit,
        pauli_string1=s1, pauli_string2=s2
    )

    # Transpile both
    print(f"\n  Transpiling for {backend.name}...")
    t0 = time.time()
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    tqc_id = pm.run(qc_identity)
    tqc_rnd = pm.run(qc_random)
    dt = time.time() - t0

    ops_id = tqc_id.count_ops()
    ops_rnd = tqc_rnd.count_ops()
    print(f"  Transpiled in {dt:.1f}s")
    print(f"  Identity circuit: depth={tqc_id.depth()}, CZ={ops_id.get('cz',0)}, "
          f"SX={ops_id.get('sx',0)}, RZ={ops_id.get('rz',0)}")
    print(f"  Random circuit:   depth={tqc_rnd.depth()}, CZ={ops_rnd.get('cz',0)}")

    # Run on QPU
    shots = 1024
    print(f"\n  Submitting 2 circuits x {shots} shots to {backend.name}...")
    print(f"  THIS USES QPU TIME. Starting timer...")

    t_qpu_start = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc_id, tqc_rnd], shots=shots)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results...")

    result = job.result()
    t_qpu_end = time.time()
    qpu_time = t_qpu_end - t_qpu_start

    print(f"\n  QPU wall time: {qpu_time:.1f}s")

    # Extract results
    counts_id = result[0].data.meas.get_counts()
    counts_rnd = result[1].data.meas.get_counts()

    # Compute <Z_i> for every qubit
    z_identity = np.array([compute_z_expectation(counts_id, n_sys, i) for i in range(n_sys)])
    z_random = np.array([compute_z_expectation(counts_rnd, n_sys, i) for i in range(n_sys)])
    delta_c4 = z_identity - z_random

    print(f"\n  RESULTS (identity - random):")
    print(f"  {'Qubit':>6s}  {'Z_id':>8s}  {'Z_rnd':>8s}  {'DeltaC4':>8s}")
    print(f"  ------  --------  --------  --------")
    for i in range(n_sys):
        marker = " M" if i == measure_qubit else (" B" if i == butterfly_qubit else "  ")
        print(f"  q{i}{marker}   {z_identity[i]:8.4f}  {z_random[i]:8.4f}  {delta_c4[i]:8.4f}")

    print(f"\n  Scalar DeltaC4 (q{measure_qubit}): {delta_c4[measure_qubit]:.6f}")
    print(f"  Mean |DeltaC4| all qubits:   {np.mean(np.abs(delta_c4)):.6f}")
    print(f"  % negative:                  {100*np.sum(delta_c4<0)/n_sys:.1f}%")

    # Save results
    output = {
        'backend': backend.name,
        'n_sys': n_sys,
        'depth': depth,
        'butterfly_qubit': butterfly_qubit,
        'measure_qubit': measure_qubit,
        'shots': shots,
        'qpu_wall_time_s': qpu_time,
        'job_id': job.job_id(),
        'transpiled_depth_id': tqc_id.depth(),
        'transpiled_cz_id': ops_id.get('cz', 0),
        'transpiled_depth_rnd': tqc_rnd.depth(),
        'transpiled_cz_rnd': ops_rnd.get('cz', 0),
        'z_identity': z_identity.tolist(),
        'z_random': z_random.tolist(),
        'delta_c4': delta_c4.tolist(),
        'counts_identity': counts_id,
        'counts_random': counts_rnd,
    }

    outfile = os.path.join(DATA_DIR, "qpu_baseline_10q.json")
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {outfile}")
    print(f"\n{'='*75}")
    print(f"  BASELINE SUMMARY")
    print(f"  QPU wall time: {qpu_time:.1f}s for 2 circuits x {shots} shots")
    print(f"  Estimated per circuit: ~{qpu_time/2:.1f}s")
    print(f"  Transpiled CZ count: {ops_id.get('cz',0)}")
    print(f"  Transpiled depth: {tqc_id.depth()}")
    print(f"{'='*75}")
    print(f"\n  PAUSED. Review results before spending more QPU time.")


if __name__ == "__main__":
    main()
