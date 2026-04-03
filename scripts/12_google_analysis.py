"""
DEEP ANALYSIS — Google's Circuits + Our Data + Real Plots
==========================================================
No half-assing. Every circuit instance. Every qubit. Every correlation.
Publication-grade topographical visualizations.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import math
import json
import os
from collections import deque
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew, pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

PI = math.pi
COS_BETA = math.cos(1/PI)
COS2_BETA = COS_BETA**2
HEX_PACK = PI / (2*math.sqrt(3))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
GOOGLE_DIR = os.path.join(REPO_ROOT, "google_circuits", "OTOC2_circuits")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_google_data():
    """Load ALL Google precomputed DeltaC4 values."""
    # fig4b has all 5 sizes
    with open(f"{GOOGLE_DIR}/fig4b_otoc2_18q_23q_27q_31q_36q/run.py") as f:
        content = f.read()

    # Extract simulated_data dict
    start = content.find("simulated_data = {")
    end = content.find("\n}", start) + 2
    exec_globals = {}
    exec(content[start:end], exec_globals)
    sim_data = exec_globals['simulated_data']

    # fig3e has 40q
    with open(f"{GOOGLE_DIR}/fig3e_otoc2_40q22c/run.py") as f:
        c40 = f.read()
    start40 = c40.find("otoc2_subtracted_precomputed = np.array(")
    end40 = c40.find("])", start40) + 2
    exec_globals40 = {'np': np}
    exec(c40[start40:end40], exec_globals40)
    sim_data[40] = exec_globals40['otoc2_subtracted_precomputed'].tolist()

    # fig4a has 65q EXPERIMENTAL
    with open(f"{GOOGLE_DIR}/fig4a_otoc2_65q23c/run.py") as f:
        c65 = f.read()
    start65 = c65.find("otoc2_subtracted_experiment = np.array(")
    end65 = c65.find("])", start65) + 2
    exec_globals65 = {'np': np}
    exec(c65[start65:end65], exec_globals65)
    sim_data[65] = exec_globals65['otoc2_subtracted_experiment'].tolist()

    return sim_data


def load_google_geometries():
    """Extract qubit subgraph geometries from circuit JSON files."""
    depths = {18: 14, 23: 16, 27: 17, 31: 18, 36: 20}
    base = f"{GOOGLE_DIR}/fig4b_otoc2_18q_23q_27q_31q_36q"
    geos = {}

    for nq, depth in depths.items():
        cfile = f"{base}/circuits_{depth}/circuit_0.pkl"
        if not os.path.exists(cfile):
            continue
        with open(cfile) as f:
            data = json.loads(f.read())

        moments = data['moments']
        all_q = set()
        for m in moments:
            for op in m.get('operations', []):
                for q in op.get('qubits', []):
                    if q.get('cirq_type') == 'GridQubit':
                        all_q.add((q['row'], q['col']))
                q2 = op.get('qubit', {})
                if q2.get('cirq_type') == 'GridQubit':
                    all_q.add((q2['row'], q2['col']))

        # Measurement qubit
        msmt = None
        for op in moments[-1].get('operations', []):
            if 'Measure' in op.get('gate', {}).get('cirq_type', op.get('cirq_type', '')):
                for q in op.get('qubits', []):
                    msmt = (q['row'], q['col'])

        # Butterfly qubits
        bfs = []
        bf_moment = moments[2 * depth]
        for op in bf_moment.get('operations', []):
            q2 = op.get('qubit', {})
            if q2.get('cirq_type') == 'GridQubit':
                bfs.append((q2['row'], q2['col']))

        # Edges
        edges = set()
        for m in moments:
            for op in m.get('operations', []):
                qs = op.get('qubits', [])
                if len(qs) == 2:
                    q1 = (qs[0]['row'], qs[0]['col'])
                    q2 = (qs[1]['row'], qs[1]['col'])
                    edges.add((min(q1, q2), max(q1, q2)))

        geos[nq] = {
            'qubits': sorted(all_q),
            'edges': sorted(edges),
            'measurement': msmt,
            'butterfly': bfs,
            'depth': depth,
            'n_qubits': len(all_q),
        }
        # Compute M-B distance
        if msmt and bfs:
            mb_dist = abs(msmt[0]-bfs[0][0]) + abs(msmt[1]-bfs[0][1])
            diam = max(abs(q1[0]-q2[0])+abs(q1[1]-q2[1]) for q1 in all_q for q2 in all_q)
            geos[nq]['mb_dist'] = mb_dist
            geos[nq]['diameter'] = diam
            geos[nq]['mb_ratio'] = mb_dist / diam

    return geos


def main():
    print("="*90)
    print("  DEEP ANALYSIS — GOOGLE DATA + IBM DATA + REAL PLOTS")
    print("="*90)

    # =====================================================================
    # PART 1: GOOGLE DATA — FULL STATISTICAL ANALYSIS
    # =====================================================================
    google = load_google_data()
    geos = load_google_geometries()

    depths = {18: 14, 23: 16, 27: 17, 31: 18, 36: 20, 40: 22, 65: 23}
    d_shifts = {18: 2, 23: 0, 27: 1, 31: 2, 36: 2, 40: 2, 65: 1}

    print(f"\n{'='*90}")
    print(f"  GOOGLE OTOC(2) — COMPLETE STATISTICAL ANALYSIS")
    print(f"{'='*90}")

    sizes = sorted(google.keys())
    stats = {}
    for n in sizes:
        vals = np.array(google[n])
        geo = geos.get(n, {})
        mb_r = geo.get('mb_ratio', 0)
        depth = depths.get(n, 0)

        s = {
            'n': n, 'depth': depth, 'count': len(vals),
            'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
            'std': float(np.std(vals)), 'abs_mean': float(np.mean(np.abs(vals))),
            'pct_neg': float(100 * np.sum(vals < 0) / len(vals)),
            'skewness': float(skew(vals)), 'kurtosis': float(kurtosis(vals)),
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
            'q25': float(np.percentile(vals, 25)),
            'q75': float(np.percentile(vals, 75)),
            'mb_ratio': mb_r, 'mb_dist': geo.get('mb_dist', 0),
            'diameter': geo.get('diameter', 0),
        }
        stats[n] = s

        print(f"\n  {n}q (depth={depth}, {len(vals)} instances, M-B/D={mb_r:.3f}):")
        print(f"    Mean DC4: {s['mean']:+.6f}  |Mean|: {s['abs_mean']:.6f}")
        print(f"    Std: {s['std']:.6f}  Skew: {s['skewness']:.3f}  Kurt: {s['kurtosis']:.3f}")
        print(f"    Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"    %neg: {s['pct_neg']:.1f}%  Q25: {s['q25']:.5f}  Q75: {s['q75']:.5f}")

    # M-B distance correction
    print(f"\n  M-B DISTANCE CORRECTION:")
    print(f"  Raw means are confounded by varying M-B geometry.")
    print(f"  Normalize: |DC4|_corrected = |DC4| / (M-B/Diam)^alpha")
    print(f"\n  {'N':>4s}  {'|DC4|':>10s}  {'MB/D':>6s}  {'Depth':>5s}  {'D/N':>5s}")
    for n in [18, 23, 27, 31, 36]:
        s = stats[n]
        print(f"  {n:4d}  {s['abs_mean']:10.6f}  {s['mb_ratio']:6.3f}  {s['depth']:5d}  {s['depth']/n:5.3f}")

    # =====================================================================
    # PART 2: BREATHING FACTOR FROM GOOGLE DATA
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"  BREATHING FACTOR EXTRACTION FROM GOOGLE DATA")
    print(f"{'='*90}")

    print(f"\n  Raw consecutive ratios (Google, no corrections):")
    goog_sizes = [18, 23, 27, 31, 36, 40, 65]
    for i in range(1, len(goog_sizes)):
        n0, n1 = goog_sizes[i-1], goog_sizes[i]
        m0, m1 = stats[n0]['abs_mean'], stats[n1]['abs_mean']
        dn = n1 - n0
        if m0 > 1e-6 and m1 > 1e-6:
            ratio = m1/m0
            per_q = ratio**(1/dn) if ratio > 0 else 0
            print(f"    {n0:3d}q -> {n1:3d}q: ratio={ratio:.4f}  per_q={per_q:.6f}  "
                  f"(cos(1/pi)={COS_BETA:.6f}, diff={abs(per_q-COS_BETA)/COS_BETA*100:.2f}%)")

    # Best pair: control for similar MB ratio
    print(f"\n  Best controlled pair (similar MB/D):")
    m18 = stats[18]['abs_mean']
    m36 = stats[36]['abs_mean']
    ratio_18_36 = m36/m18
    per_q_18_36 = ratio_18_36**(1/18)
    print(f"    18q->36q (MB/D: 0.714->0.636): per_q = {per_q_18_36:.6f}")
    print(f"    cos(1/pi) = {COS_BETA:.6f}")
    print(f"    Difference: {abs(per_q_18_36-COS_BETA)/COS_BETA*100:.2f}%")

    # Overall 18->65 (largest span)
    m65 = stats[65]['abs_mean']
    ratio_18_65 = m65/m18
    per_q_18_65 = ratio_18_65**(1/47)
    print(f"\n    18q->65q (full Google span): per_q = {per_q_18_65:.6f}")
    print(f"    cos(1/pi) = {COS_BETA:.6f}")
    print(f"    Difference: {abs(per_q_18_65-COS_BETA)/COS_BETA*100:.2f}%")

    # =====================================================================
    # LOAD OUR IBM DATA
    # =====================================================================
    ibm_data = {}
    for fn in ['qpu_fullsweep.json', 'listen_echoes.json', 'sphere_s2n3.json',
               'butterfly_averaged.json', 'qpu_full_experiment.json']:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            with open(path) as f:
                ibm_data[fn] = json.load(f)

    # =====================================================================
    # PART 3: REAL PLOTS
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"  GENERATING REAL PLOTS")
    print(f"{'='*90}")

    # PLOT 1: Google data — distribution per size
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('Google OTOC(2) $\\Delta C^{(4)}$ — Raw Distributions (Willow Square Lattice)',
                 fontsize=16, color='white', y=0.98)

    for idx, n in enumerate([18, 23, 27, 31, 36, 40, 65]):
        ax = axes.flat[idx]
        vals = np.array(google[n])
        geo = geos.get(n, {})

        ax.hist(vals, bins=25, color='#4ecdc4', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(vals), color='yellow', linestyle='-', linewidth=2, label=f'mean={np.mean(vals):.4f}')

        ax.set_title(f'{n}q (d={depths[n]}, MB/D={geo.get("mb_ratio",0):.2f})',
                     color='white', fontsize=11)
        ax.set_xlabel('$\\Delta C^{(4)}$', color='white', fontsize=9)
        ax.set_ylabel('Count', color='white', fontsize=9)
        ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#333')

    # Last panel: summary
    ax = axes.flat[7]
    ns = [18, 23, 27, 31, 36, 40, 65]
    means = [stats[n]['abs_mean'] for n in ns]
    stds = [stats[n]['std'] for n in ns]
    ax.errorbar(ns, means, yerr=stds, fmt='o-', color='cyan', markersize=8,
                capsize=5, ecolor='gray', elinewidth=1)
    ax.set_xlabel('Qubits', color='white', fontsize=11)
    ax.set_ylabel('Mean |$\\Delta C^{(4)}$|', color='white', fontsize=11)
    ax.set_title('Size Scaling (error = 1 std)', color='white', fontsize=11)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, 'google_distributions.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"  google_distributions.png")

    # PLOT 2: Google subgraph geometries
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor('#0a0a1a')
    fig.suptitle('Google Willow Subgraph Geometries — Qubit Positions on Square Lattice',
                 fontsize=16, color='white')

    for idx, n in enumerate([18, 23, 27, 31, 36]):
        ax = axes[idx]
        geo = geos[n]
        qubits = geo['qubits']
        edges = geo['edges']
        msmt = geo['measurement']
        bfs = geo['butterfly']

        for q1, q2 in edges:
            ax.plot([q1[1], q2[1]], [q1[0], q2[0]], 'gray', linewidth=0.5, alpha=0.3)

        for q in qubits:
            color = '#ff4444' if q == msmt else ('#44ff44' if q in bfs else '#4488ff')
            size = 120 if q == msmt or q in bfs else 60
            ax.scatter(q[1], q[0], c=color, s=size, zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_title(f'{n}q  d={geo["depth"]}  MB/D={geo["mb_ratio"]:.2f}',
                     color='white', fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values(): spine.set_color('#333')

    axes[0].scatter([], [], c='#ff4444', s=80, label='Measure', edgecolors='white')
    axes[0].scatter([], [], c='#44ff44', s=80, label='Butterfly', edgecolors='white')
    axes[0].scatter([], [], c='#4488ff', s=40, label='Other', edgecolors='white')
    axes[0].legend(fontsize=8, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'google_geometries.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"  google_geometries.png")

    # PLOT 3: Combined Google + IBM scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0a0a1a')

    gn = [18, 23, 27, 31, 36, 40, 65]
    gm = [stats[n]['abs_mean'] for n in gn]
    gs = [stats[n]['std']/math.sqrt(stats[n]['count']) for n in gn]
    gmb = [geos.get(n, {}).get('mb_ratio', 0) for n in gn]

    ax1.errorbar(gn, gm, yerr=gs, fmt='s', color='#ff6b6b', markersize=10,
                 capsize=6, ecolor='#ff6b6b', elinewidth=2, label='Google Willow (sim+exp)')
    for i, n in enumerate(gn):
        ax1.annotate(f'MB={gmb[i]:.2f}', (n, gm[i]), textcoords="offset points",
                     xytext=(5, 10), fontsize=7, color='#ff9999')

    if 'qpu_fullsweep.json' in ibm_data:
        sweep = ibm_data['qpu_fullsweep.json']
        valid = [r for r in sweep['results'] if r['cz_gates'] >= r['n_sys']]
        in_ = [r['n_sys'] for r in valid]
        im = [r['dc4_mean_abs'] for r in valid]
        ax1.scatter(in_, im, c='#4ecdc4', s=120, zorder=5, edgecolors='white',
                    linewidth=2, label='IBM Heron (raw hardware)')

    n_fit = np.linspace(10, 160, 200)
    for A, ls in [(0.9, '-'), (0.5, '--'), (0.2, ':')]:
        ax1.plot(n_fit, A * COS_BETA**n_fit, color='yellow', linestyle=ls,
                 alpha=0.4, linewidth=1)
    ax1.plot([], [], color='yellow', linestyle='-', alpha=0.4, label=f'cos(1/$\\pi$)$^N$ reference')

    ax1.set_xlabel('System Size (qubits)', fontsize=13, color='white')
    ax1.set_ylabel('Mean |$\\Delta C^{(4)}$|', fontsize=13, color='white')
    ax1.set_title('OTOC(2) Echo: Google Willow vs IBM Heron\nDifferent topology, same physics?', fontsize=13, color='white')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15)
    for spine in ax1.spines.values(): spine.set_color('#333')

    ax2.axhline(y=COS_BETA, color='red', linestyle='-', linewidth=2, alpha=0.8,
                label=f'cos(1/$\\pi$) = {COS_BETA:.4f}')
    ax2.axhspan(COS_BETA*0.99, COS_BETA*1.01, color='red', alpha=0.1, label='1% band')

    for i in range(1, len(gn)):
        n0, n1 = gn[i-1], gn[i]
        m0, m1 = gm[i-1], gm[i]
        dn = n1 - n0
        if m0 > 1e-6 and m1 > 1e-6:
            ratio = m1/m0
            per_q = ratio**(1/dn) if ratio > 0 else 1
            mid_n = (n0 + n1) / 2
            ax2.scatter(mid_n, per_q, c='#ff6b6b', s=80, zorder=5, edgecolors='white')
            ax2.annotate(f'{n0}-{n1}', (mid_n, per_q), textcoords="offset points",
                         xytext=(5, 5), fontsize=7, color='#ff9999')

    if 'qpu_fullsweep.json' in ibm_data:
        valid = [r for r in ibm_data['qpu_fullsweep.json']['results'] if r['cz_gates'] >= r['n_sys']]
        for i in range(1, len(valid)):
            r0, r1 = valid[i-1], valid[i]
            m0, m1 = r0['dc4_mean_abs'], r1['dc4_mean_abs']
            dn = r1['n_sys'] - r0['n_sys']
            if m0 > 1e-4 and m1 > 1e-4 and dn > 0:
                ratio = m1/m0
                per_q = ratio**(1/dn) if ratio > 0 else 1
                mid_n = (r0['n_sys'] + r1['n_sys']) / 2
                ax2.scatter(mid_n, per_q, c='#4ecdc4', s=100, zorder=5, edgecolors='white',
                            marker='D')

    ax2.scatter([], [], c='#ff6b6b', s=60, label='Google pairs')
    ax2.scatter([], [], c='#4ecdc4', s=60, marker='D', label='IBM pairs')
    ax2.set_xlabel('Midpoint (qubits)', fontsize=13, color='white')
    ax2.set_ylabel('Per-qubit decay rate', fontsize=13, color='white')
    ax2.set_title('Per-Qubit Echo Decay Rate\nDoes it converge to cos(1/$\\pi$)?', fontsize=13, color='white')
    ax2.set_ylim(0.4, 1.6)
    ax2.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15)
    for spine in ax2.spines.values(): spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'google_vs_ibm_scaling.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"  google_vs_ibm_scaling.png")

    print(f"\n  All plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
