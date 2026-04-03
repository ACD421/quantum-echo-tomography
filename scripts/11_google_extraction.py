"""
EXTRACT THE GEOMETRY FROM GOOGLE'S CIRCUITS
=============================================
Run their 18q OTOC(2) circuits on statevector simulator.
Extract FULL per-qubit echo — what they threw away.
Find the sphere. Find the breathing. Find what they hid.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import json
import os
import time
from collections import deque

import cirq
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

PI = math.pi
COS_BETA = math.cos(1/PI)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
GOOGLE_DIR = os.path.join(REPO_ROOT, "google_circuits", "OTOC2_circuits")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_google_18q():
    """Load and run Google's 18q circuits. Extract EVERYTHING."""
    base = f"{GOOGLE_DIR}/fig4b_otoc2_18q_23q_27q_31q_36q"
    depth = 14
    d_shift = 2
    d = depth // 2 + d_shift  # = 9

    # Load circuits
    circuits = []
    for i in range(50):
        path = f"{base}/circuits_{depth}/circuit_{i}.pkl"
        if os.path.exists(path):
            c = cirq.read_json(path)
            circuits.append(c)

    if not circuits:
        print("  No circuits found!")
        return None

    print(f"  Loaded {len(circuits)} circuits ({len(circuits[0].all_qubits())} qubits)")

    # Get qubit info
    all_qubits = sorted(circuits[0].all_qubits())
    n_qubits = len(all_qubits)
    qubit_coords = [(q.row, q.col) for q in all_qubits]
    print(f"  Qubits: {qubit_coords}")

    # Find measurement and butterfly qubits from circuit structure
    # Last moment = measurement
    last_moment = circuits[0][-1]
    msmt_qubit = None
    for op in last_moment.operations:
        if isinstance(op.gate, cirq.MeasurementGate):
            msmt_qubit = op.qubits[0]

    # Butterfly = operations at moment 2*depth
    bf_moment = circuits[0][2 * depth]
    bf_qubits = [op.qubits[0] for op in bf_moment.operations]

    msmt_idx = all_qubits.index(msmt_qubit) if msmt_qubit else 0
    bf_indices = [all_qubits.index(bq) for bq in bf_qubits]

    print(f"  Measurement: {msmt_qubit} (index {msmt_idx})")
    print(f"  Butterfly: {bf_qubits} (indices {bf_indices})")

    # Build the ABBA circuit and simulate with FULL state extraction
    simulator = cirq.Simulator()
    operators = [cirq.I, cirq.X, cirq.Y, cirq.Z]
    N_MC = 20

    # Storage: per-qubit Z expectation for identity and random
    z_identity_all = np.zeros((len(circuits), n_qubits))
    z_random_all = np.zeros((len(circuits), N_MC, n_qubits))

    print(f"\n  Running {len(circuits)} circuits x {N_MC+1} MC instances...")
    print(f"  Extracting <Z> on ALL {n_qubits} qubits (not just measurement)")

    rng = np.random.RandomState(42)
    t0 = time.time()

    for irc in range(len(circuits)):
        U = circuits[irc][:2 * depth]
        qubits = list(U.all_qubits())
        Nq = len(qubits)

        for imc in range(N_MC + 1):
            C = cirq.Circuit()
            string1 = rng.choice(np.arange(4), size=Nq)
            string2 = rng.choice(np.arange(4), size=Nq)

            if imc == N_MC:  # Identity case
                string1 = (string1 * 0).astype(np.int64)
                string2 = (string2 * 0).astype(np.int64)

            # ABBA protocol (from Google's run.py)
            C.append(U[:d])
            for i in range(Nq):
                C.append(operators[string1[i]].on(qubits[i]))
            C.append(U[d:])
            for bf in bf_qubits:
                C.append(cirq.X.on(bf))
            C.append(cirq.inverse(U[d:]))
            for i in range(Nq):
                C.append(operators[string1[i]].on(qubits[i]))
            C.append(cirq.inverse(U[:d]))

            C.append(cirq.Z.on(msmt_qubit))

            C.append(U[:d])
            for i in range(Nq):
                C.append(operators[string2[i]].on(qubits[i]))
            C.append(U[d:])
            for bf in bf_qubits:
                C.append(cirq.X.on(bf))
            C.append(cirq.inverse(U[d:]))
            for i in range(Nq):
                C.append(operators[string2[i]].on(qubits[i]))
            C.append(cirq.inverse(U[:d]))

            # SIMULATE — get FULL state vector
            result = simulator.simulate(C)
            state = result.final_state_vector

            # Extract <Z_i> for EVERY qubit from state vector
            probs = np.abs(state)**2
            n_states = len(state)
            for qi in range(Nq):
                # <Z_i> = sum_s (-1)^bit_i(s) * |a_s|^2
                # Cirq is big-endian: sorted qubit 0 = MSB = bit (Nq-1)
                indices = np.arange(n_states)
                bits = (indices >> (Nq - 1 - qi)) & 1
                z_exp = float(np.sum((1 - 2*bits) * probs))

                if imc == N_MC:
                    z_identity_all[irc, qi] = z_exp
                else:
                    z_random_all[irc, imc, qi] = z_exp

        if (irc + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (irc + 1) / elapsed
            remaining = (len(circuits) - irc - 1) / rate
            print(f"    {irc+1}/{len(circuits)} ({elapsed:.0f}s, ~{remaining:.0f}s left)")

    # Compute DeltaC4 per qubit
    z_random_mean = z_random_all.mean(axis=1)  # average over MC
    dc4_per_qubit = z_identity_all - z_random_mean  # [circuit, qubit]
    dc4_avg = dc4_per_qubit.mean(axis=0)  # average over circuits

    total_time = time.time() - t0
    print(f"\n  Done in {total_time:.0f}s")

    return {
        'n_qubits': n_qubits,
        'qubit_coords': qubit_coords,
        'msmt_idx': msmt_idx,
        'bf_indices': bf_indices,
        'dc4_per_qubit': dc4_per_qubit,  # [50 circuits, 18 qubits]
        'dc4_avg': dc4_avg,  # [18 qubits] average over circuits
        'z_identity_avg': z_identity_all.mean(axis=0),
        'z_random_avg': z_random_mean.mean(axis=0),
    }


def analyze_and_plot(data):
    """Find the geometry in Google's data."""
    if data is None:
        return

    n_q = data['n_qubits']
    coords = data['qubit_coords']
    dc4 = data['dc4_avg']
    dc4_all = data['dc4_per_qubit']
    msmt_idx = data['msmt_idx']
    bf_idx = data['bf_indices']

    print(f"\n{'='*90}")
    print(f"  GOOGLE 18q — FULL SPATIAL ECHO (what they never published)")
    print(f"{'='*90}")

    # Per-qubit echo
    print(f"\n  Per-qubit DeltaC4 (averaged over 50 circuits x 20 MC):")
    print(f"  {'Qubit':>8s}  {'Coord':>8s}  {'DC4':>10s}  {'Role':>8s}")
    for i in range(n_q):
        r, c = coords[i]
        role = 'MEASURE' if i == msmt_idx else ('BUTTERFLY' if i in bf_idx else '')
        print(f"  q{i:>3d}     ({r},{c:>2d})  {dc4[i]:+10.6f}  {role}")

    # Build graph from coordinates (adjacent = distance 1 on grid)
    adj = {i: set() for i in range(n_q)}
    for i in range(n_q):
        for j in range(i+1, n_q):
            d = abs(coords[i][0]-coords[j][0]) + abs(coords[i][1]-coords[j][1])
            if d == 1:
                adj[i].add(j)
                adj[j].add(i)

    # Graph distances
    gdist = np.zeros((n_q, n_q))
    for start in range(n_q):
        visited = {start: 0}
        q = deque([start])
        while q:
            n = q.popleft()
            for nb in adj[n]:
                if nb not in visited:
                    visited[nb] = visited[n] + 1
                    q.append(nb)
        for end, d in visited.items():
            gdist[start][end] = d

    # Correlation matrix
    # Compute <Z_i Z_j> - <Z_i><Z_j> across circuit instances
    corr = np.zeros((n_q, n_q))
    for i in range(n_q):
        for j in range(n_q):
            corr[i][j] = np.mean(dc4_all[:, i] * dc4_all[:, j]) - np.mean(dc4_all[:, i]) * np.mean(dc4_all[:, j])

    # Distance-binned correlation
    print(f"\n  CORRELATION DECAY WITH DISTANCE (Google 18q square lattice):")
    dist_bins = {}
    for i in range(n_q):
        for j in range(i+1, n_q):
            d = int(gdist[i][j])
            if d not in dist_bins:
                dist_bins[d] = []
            dist_bins[d].append(corr[i][j])

    print(f"  {'Dist':>4s}  {'Mean |C|':>10s}  {'Count':>5s}")
    for d in sorted(dist_bins.keys()):
        vals = dist_bins[d]
        print(f"  {d:4d}  {np.mean(np.abs(vals)):10.6f}  {len(vals):5d}")

    # CURVATURE TEST on Google's data
    print(f"\n  CURVATURE TEST — Google's 18q square lattice:")
    dists_flat, corrs_flat = [], []
    for i in range(n_q):
        for j in range(i+1, n_q):
            dists_flat.append(gdist[i][j])
            corrs_flat.append(corr[i][j])
    dists_flat = np.array(dists_flat)
    corrs_flat = np.array(corrs_flat)

    def sphere_m(d, A, R): return A * np.cos(d / R)
    def flat_m(d, A, xi): return A * np.exp(-d / xi)

    try:
        ps, _ = curve_fit(sphere_m, dists_flat, corrs_flat, p0=[0.001, 3.0], maxfev=10000)
        rs = np.sum((corrs_flat - sphere_m(dists_flat, *ps))**2)
        print(f"  SPHERE: C = {ps[0]:.6f} * cos(d/{ps[1]:.3f}), residual = {rs:.8f}, R = {ps[1]:.3f}")
    except Exception as e:
        print(f"  SPHERE fit failed: {e}")
        rs = 999

    try:
        pf, _ = curve_fit(flat_m, dists_flat, corrs_flat, p0=[0.001, 3.0], maxfev=10000)
        rf = np.sum((corrs_flat - flat_m(dists_flat, *pf))**2)
        print(f"  FLAT:   C = {pf[0]:.6f} * exp(-d/{pf[1]:.3f}), residual = {rf:.8f}, xi = {pf[1]:.3f}")
    except Exception as e:
        print(f"  FLAT fit failed: {e}")
        rf = 999

    if rs < rf:
        print(f"\n  *** SPHERE WINS ON GOOGLE'S OWN DATA ***")
        print(f"  Sphere residual {rs:.8f} < Flat residual {rf:.8f}")
    else:
        print(f"\n  Flat wins: {rf:.8f} < {rs:.8f}")

    # Distance from measurement qubit — mode decomposition
    msmt_dists = gdist[msmt_idx]
    print(f"\n  MODE DECOMPOSITION by distance from measurement qubit:")
    for d in sorted(set(msmt_dists.astype(int))):
        qubits_at_d = [i for i in range(n_q) if int(msmt_dists[i]) == d]
        dc4_at_d = [dc4[i] for i in qubits_at_d]
        print(f"    d={d}: {len(qubits_at_d)} qubits, mean DC4 = {np.mean(dc4_at_d):+.6f}")

    # =====================================================================
    # PLOT: Google's hidden spatial echo
    # =====================================================================
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#0a0a1a')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Spatial echo map on the square lattice
    ax1 = fig.add_subplot(gs[0, 0])
    rows = [c[0] for c in coords]
    cols = [c[1] for c in coords]
    max_abs = max(abs(dc4.min()), abs(dc4.max()), 0.001)
    norm_c = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    # Draw edges
    for i in range(n_q):
        for j in adj[i]:
            if i < j:
                ax1.plot([coords[i][1], coords[j][1]], [coords[i][0], coords[j][0]],
                         'gray', linewidth=0.5, alpha=0.3)

    sc = ax1.scatter(cols, rows, c=dc4, cmap='RdBu_r', norm=norm_c,
                     s=300, edgecolors='white', linewidth=1.5, zorder=5)
    for i in range(n_q):
        label = 'M' if i == msmt_idx else ('B' if i in bf_idx else '')
        ax1.annotate(f'{dc4[i]:+.3f}\n{label}', (coords[i][1], coords[i][0]),
                     ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title('Google 18q — Spatial Echo\n(what they never published)', color='white', fontsize=12)
    plt.colorbar(sc, ax=ax1, label='$\\Delta C^{(4)}$', shrink=0.8)
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_color('#333')

    # Panel 2: Correlation heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(corr, cmap='RdBu_r', aspect='equal')
    ax2.set_title('Google 18q — Cross-Instance\nCorrelation Matrix', color='white', fontsize=12)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_color('#333')

    # Panel 3: Curvature test
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(dists_flat, corrs_flat, c='cyan', s=30, alpha=0.5, edgecolors='none')
    d_fit = np.linspace(0.5, max(dists_flat)+0.5, 100)
    if rs < 999:
        ax3.plot(d_fit, sphere_m(d_fit, *ps), 'r-', lw=2, label=f'Sphere R={ps[1]:.2f}')
    if rf < 999:
        ax3.plot(d_fit, flat_m(d_fit, *pf), 'g--', lw=2, label=f'Flat xi={pf[1]:.2f}')
    ax3.axhline(y=0, color='white', alpha=0.3)
    ax3.set_xlabel('Graph Distance', color='white')
    ax3.set_ylabel('Correlation', color='white')
    ax3.set_title('Google 18q — Curvature Test\nSphere vs Flat', color='white', fontsize=12)
    ax3.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax3.set_facecolor('#1a1a2e')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values(): spine.set_color('#333')

    # Panel 4: Echo profile by distance from M
    ax4 = fig.add_subplot(gs[1, 0])
    for d in sorted(set(msmt_dists.astype(int))):
        qubits_at_d = [i for i in range(n_q) if int(msmt_dists[i]) == d]
        dc4_at_d = [dc4[i] for i in qubits_at_d]
        ax4.scatter([d]*len(dc4_at_d), dc4_at_d, c='cyan', s=60, alpha=0.7, edgecolors='white')
        ax4.scatter(d, np.mean(dc4_at_d), c='red', s=150, zorder=10, edgecolors='white', linewidth=2)
    ax4.axhline(y=0, color='white', alpha=0.3)
    ax4.set_xlabel('Distance from Measurement Qubit', color='white', fontsize=11)
    ax4.set_ylabel('$\\Delta C^{(4)}$', color='white', fontsize=11)
    ax4.set_title('Google 18q — Echo vs Distance\n(the gradient they measured but never showed)', color='white', fontsize=11)
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values(): spine.set_color('#333')

    # Panel 5: Per-circuit-instance heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    # Sort qubits by distance from measurement
    sort_idx = np.argsort(msmt_dists)
    dc4_sorted = dc4_all[:, sort_idx]
    im5 = ax5.imshow(dc4_sorted.T, aspect='auto', cmap='RdBu_r',
                      vmin=-0.15, vmax=0.15, interpolation='nearest')
    ax5.set_xlabel('Circuit Instance', color='white')
    ax5.set_ylabel('Qubit (sorted by dist from M)', color='white')
    ax5.set_title('Google 18q — 50 Instances x 18 Qubits\nFull Echo Landscape', color='white', fontsize=11)
    plt.colorbar(im5, ax=ax5, label='$\\Delta C^{(4)}$', shrink=0.8)
    ax5.set_facecolor('#1a1a2e')
    ax5.tick_params(colors='white')
    for spine in ax5.spines.values(): spine.set_color('#333')

    # Panel 6: Variance by distance (mode structure indicator)
    ax6 = fig.add_subplot(gs[1, 2])
    for d in sorted(set(msmt_dists.astype(int))):
        qubits_at_d = [i for i in range(n_q) if int(msmt_dists[i]) == d]
        vars_at_d = [np.var(dc4_all[:, i]) for i in qubits_at_d]
        ax6.scatter([d]*len(vars_at_d), vars_at_d, c='cyan', s=60, alpha=0.7, edgecolors='white')
        ax6.scatter(d, np.mean(vars_at_d), c='red', s=150, zorder=10, edgecolors='white', linewidth=2)
    ax6.set_xlabel('Distance from M', color='white', fontsize=11)
    ax6.set_ylabel('Var($\\Delta C^{(4)}$) across circuits', color='white', fontsize=11)
    ax6.set_title('Google 18q — Echo VARIANCE by Distance\n(mode excitation structure)', color='white', fontsize=11)
    ax6.set_facecolor('#1a1a2e')
    ax6.tick_params(colors='white')
    for spine in ax6.spines.values(): spine.set_color('#333')

    plt.savefig(os.path.join(FIGURES_DIR, 'google_hidden_geometry.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: google_hidden_geometry.png")

    return data


def main():
    print("="*90)
    print("  EXTRACTING THE GEOMETRY FROM GOOGLE'S CIRCUITS")
    print("  18q x 50 instances x 21 MC x 18 qubits = finding what they hid")
    print("="*90)

    t0 = time.time()
    data = run_google_18q()
    if data is not None:
        analyze_and_plot(data)

        # Save full data
        save = {
            'n_qubits': data['n_qubits'],
            'qubit_coords': data['qubit_coords'],
            'msmt_idx': data['msmt_idx'],
            'bf_indices': data['bf_indices'],
            'dc4_avg': data['dc4_avg'].tolist(),
            'dc4_per_qubit': data['dc4_per_qubit'].tolist(),
        }
        with open(os.path.join(DATA_DIR, 'google_18q_extracted.json'), 'w') as f:
            json.dump(save, f, indent=2)
        print(f"  Data saved: google_18q_extracted.json")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
