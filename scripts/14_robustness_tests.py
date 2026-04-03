"""
Robustness analysis for OTOC Dirac vs Laplacian decomposition.
Three analyses: permutation test, distance-from-butterfly, effect size + CI.
"""
import json
import numpy as np
from scipy import stats
from itertools import combinations
import os

np.random.seed(42)

# ── Load data ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

with open(os.path.join(DATA_DIR, "dirac_experiment.json")) as f:
    exp = json.load(f)

with open(os.path.join(DATA_DIR, "dirac_replication.json")) as f:
    rep = json.load(f)

with open(os.path.join(DATA_DIR, "patch_c_result.json")) as f:
    pc = json.load(f)

# ── Patch definitions ────────────────────────────────────────────────────────
# All use LOCAL indices (position in qubit list)
# Patch A: qubits [0,1,2,3,4,5,6,16,23] → local [0,1,2,3,4,5,6,7,8]
PATCH_A = {
    'name': 'Patch A (hub q3)',
    'qubits': [0, 1, 2, 3, 4, 5, 6, 16, 23],
    'center': 3,  # physical
    'edges': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(3,16),(16,23)],
    'dirac_j05_phys': [3, 0],
    'dirac_j15_phys': [2, 4, 16, 5],
    'dirac_gauge_phys': [1, 23, 6],
    'butterfly_phys': 3,
}

PATCH_B = {
    'name': 'Patch B (hub q7)',
    'qubits': [4, 5, 6, 7, 8, 9, 10, 17, 27],
    'center': 7,
    'edges': [(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(7,17),(17,27)],
    'dirac_j05_phys': [7, 4],
    'dirac_j15_phys': [6, 8, 17, 9],
    'dirac_gauge_phys': [5, 27, 10],
    'butterfly_phys': 7,
}

PATCH_C = {
    'name': 'Patch C (hub q151)',
    'qubits': [151, 138, 150, 152, 131, 149, 153, 130, 132],
    'center': 151,
    'edges': [(130,131),(131,132),(131,138),(138,151),(149,150),(150,151),(151,152),(152,153)],
    'dirac_j05_phys': [151, 130],
    'dirac_j15_phys': [138, 150, 152, 131],
    'dirac_gauge_phys': [149, 153, 132],
    'butterfly_phys': 151,
}

def phys_to_local(patch):
    """Convert physical qubit IDs to local indices."""
    ptol = {pq: i for i, pq in enumerate(patch['qubits'])}
    return ptol

def get_local_assignments(patch):
    """Return (j05_local, j15_local, gauge_local) index lists."""
    ptol = phys_to_local(patch)
    j05 = [ptol[q] for q in patch['dirac_j05_phys']]
    j15 = [ptol[q] for q in patch['dirac_j15_phys']]
    gauge = [ptol[q] for q in patch['dirac_gauge_phys']]
    return j05, j15, gauge

def compute_dirac_r2(dc4_vec, j05_idx, j15_idx, gauge_idx):
    """
    Compute Dirac R² for a given DC4 vector and qubit grouping.
    R² = 1 - (residual SS / total SS)
    where the model assigns each qubit its group mean.
    """
    dc4 = np.array(dc4_vec, dtype=float)
    total_mean = np.mean(dc4)
    ss_tot = np.sum((dc4 - total_mean)**2)
    if ss_tot == 0:
        return 0.0

    predicted = np.zeros_like(dc4)
    for idx_set in [j05_idx, j15_idx, gauge_idx]:
        group_mean = np.mean(dc4[idx_set])
        for i in idx_set:
            predicted[i] = group_mean
    ss_res = np.sum((dc4 - predicted)**2)
    return 1.0 - ss_res / ss_tot


# ── Extract per-seed DC4 vectors ─────────────────────────────────────────────
# Patch A depth=6 is in dirac_experiment.json results[1]
patch_a_seeds_dc4 = []
for sr in exp['results'][1]['seed_results']:
    patch_a_seeds_dc4.append(sr['dc4'])

# Patch B seeds are in dirac_replication.json patch_b_seeds
patch_b_seeds_dc4 = []
for sr in rep['patch_b_seeds']:
    # Reconstruct DC4 from decomposition? No - need raw DC4.
    # The replication file has per-seed decompositions but not raw dc4 for B.
    pass

# Actually let me check: patch_b in replication has dir_r2s but not raw dc4.
# The experiment file results[1] is depth=6 which is Patch A.
# dirac_replication has patch_a (which reuses the depth=6 data) and patch_b.
# For Patch B, the raw DC4 per seed is NOT stored - only decompositions + R2s.
# We can still do analyses 1 and 3 using the stored R² values.
# For analysis 1 (permutation test), we NEED raw DC4 vectors.

# Let me check what's actually available more carefully.
# Patch A: exp['results'][1]['seed_results'][i]['dc4'] - YES, 10 seeds x 9 values
# Patch B: rep has dir_r2s and lap_r2s (10 seeds) but no raw dc4
# Patch C: pc['all_dc4'] - YES, 5 seeds x 9 values

# For Patch B, we only have the R² values, not raw DC4.
# We CAN still do analysis 1 if we note that the Patch A depth=6 data
# IS the same as rep['patch_a'] (same R² values match).
# For Patch B we'll have to skip the permutation test and note it.

print("=" * 72)
print("ROBUSTNESS ANALYSIS: OTOC Dirac vs Laplacian Decomposition")
print("=" * 72)

# Collect data
patches_data = []

# Patch A (depth=6, 10 seeds)
a_j05, a_j15, a_gauge = get_local_assignments(PATCH_A)
a_dc4s = [sr['dc4'] for sr in exp['results'][1]['seed_results']]
a_dir_r2s = np.array([sr['dirac_r2'] for sr in exp['results'][1]['seed_results']])
a_lap_r2s = np.array([sr['lap_r2'] for sr in exp['results'][1]['seed_results']])
patches_data.append(('Patch A', PATCH_A, a_dc4s, a_dir_r2s, a_lap_r2s, a_j05, a_j15, a_gauge))

# Patch B (10 seeds, but only R² available, no raw DC4)
b_dir_r2s = np.array(rep['patch_b']['dir_r2s'])
b_lap_r2s = np.array(rep['patch_b']['lap_r2s'])
# No raw DC4 for permutation test or distance analysis

# Patch C (5 seeds)
c_j05, c_j15, c_gauge = get_local_assignments(PATCH_C)
c_dc4s = pc['all_dc4']  # 5 seeds x 9 values
c_dir_r2s = np.array(pc['patch_c']['dir_r2s'])
c_lap_r2s = np.array(pc['patch_c']['lap_r2s'])
patches_data.append(('Patch C', PATCH_C, c_dc4s, c_dir_r2s, c_lap_r2s, c_j05, c_j15, c_gauge))


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ANALYSIS 1: PERMUTATION TEST (10,000 random 2+4+3 splits)")
print("=" * 72)
print("H0: Any random 2+4+3 partition yields R² as high as the Dirac assignment.")
print("If p < 0.05, the specific Dirac grouping is significantly better than chance.\n")

N_PERM = 10000

def permutation_test(dc4_list, true_j05, true_j15, true_gauge, n_perm=N_PERM):
    """
    For each seed's DC4 vector, randomly permute which qubits go into
    j=1/2 (2), j=3/2 (4), gauge (3). Compute Dirac R² for each permutation.
    Return: per-seed p-values and overall p-value.
    """
    n_qubits = 9
    indices = list(range(n_qubits))
    n_seeds = len(dc4_list)

    # True R² per seed
    true_r2s = []
    for dc4 in dc4_list:
        r2 = compute_dirac_r2(dc4, true_j05, true_j15, true_gauge)
        true_r2s.append(r2)

    # Permutation distribution
    perm_r2s = np.zeros((n_seeds, n_perm))
    rng = np.random.RandomState(12345)

    for p in range(n_perm):
        perm = rng.permutation(n_qubits)
        pj05 = list(perm[:2])
        pj15 = list(perm[2:6])
        pgauge = list(perm[6:9])

        for s, dc4 in enumerate(dc4_list):
            perm_r2s[s, p] = compute_dirac_r2(dc4, pj05, pj15, pgauge)

    # Per-seed p-values
    seed_pvals = []
    for s in range(n_seeds):
        p_val = np.mean(perm_r2s[s] >= true_r2s[s])
        seed_pvals.append(p_val)

    # Combined: for each permutation, compute mean R² across seeds
    mean_perm_r2 = np.mean(perm_r2s, axis=0)
    mean_true_r2 = np.mean(true_r2s)
    combined_p = np.mean(mean_perm_r2 >= mean_true_r2)

    return true_r2s, seed_pvals, combined_p, perm_r2s

# Patch A
print("--- Patch A (hub q3, 10 seeds, depth=6) ---")
a_true_r2, a_seed_pvals, a_combined_p, a_perm_dist = permutation_test(
    a_dc4s, a_j05, a_j15, a_gauge)

for i, (r2, pv) in enumerate(zip(a_true_r2, a_seed_pvals)):
    print(f"  Seed {i}: Dirac R² = {r2:.4f}, p = {pv:.4f}")
print(f"  Combined (mean R² across seeds): actual = {np.mean(a_true_r2):.4f}, p = {a_combined_p:.4f}")
mean_perm_a = np.mean(np.mean(a_perm_dist, axis=0))
std_perm_a = np.std(np.mean(a_perm_dist, axis=0))
print(f"  Permutation null: mean R² = {mean_perm_a:.4f} ± {std_perm_a:.4f}")
print(f"  Actual mean R² / null mean R² = {np.mean(a_true_r2)/mean_perm_a:.2f}x")

# Patch C
print("\n--- Patch C (hub q151, 5 seeds) ---")
c_true_r2, c_seed_pvals, c_combined_p, c_perm_dist = permutation_test(
    c_dc4s, c_j05, c_j15, c_gauge)

for i, (r2, pv) in enumerate(zip(c_true_r2, c_seed_pvals)):
    print(f"  Seed {i}: Dirac R² = {r2:.4f}, p = {pv:.4f}")
print(f"  Combined (mean R² across seeds): actual = {np.mean(c_true_r2):.4f}, p = {c_combined_p:.4f}")
mean_perm_c = np.mean(np.mean(c_perm_dist, axis=0))
std_perm_c = np.std(np.mean(c_perm_dist, axis=0))
print(f"  Permutation null: mean R² = {mean_perm_c:.4f} ± {std_perm_c:.4f}")
print(f"  Actual mean R² / null mean R² = {np.mean(c_true_r2)/mean_perm_c:.2f}x")

print("\n  [Note: Patch B raw DC4 vectors not stored in data files;")
print("   permutation test cannot be run. R² values available for Analysis 3.]")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: DISTANCE-FROM-BUTTERFLY CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ANALYSIS 2: DISTANCE-FROM-BUTTERFLY (radial decay) MODEL")
print("=" * 72)
print("Tests whether DC4 is simply a function of graph distance from the")
print("butterfly qubit. If distance R2 >= Dirac R2, the mode structure may")
print("just be radial decay.\n")

def compute_graph_distances(qubits, edges, butterfly_phys):
    """BFS from butterfly qubit on physical graph, return dict {phys_qubit: distance}."""
    from collections import deque
    adj = {q: [] for q in qubits}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)

    dist = {butterfly_phys: 0}
    queue = deque([butterfly_phys])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist

def distance_model_r2(dc4_vec, distances_local):
    """Fit DC4 = a + b*distance via OLS, return R²."""
    dc4 = np.array(dc4_vec, dtype=float)
    d = np.array(distances_local, dtype=float)

    # Simple OLS: DC4 = a + b*d
    n = len(dc4)
    d_mean = np.mean(d)
    dc4_mean = np.mean(dc4)

    ss_dd = np.sum((d - d_mean)**2)
    if ss_dd == 0:
        return 0.0

    b = np.sum((d - d_mean) * (dc4 - dc4_mean)) / ss_dd
    a = dc4_mean - b * d_mean

    predicted = a + b * d
    ss_res = np.sum((dc4 - predicted)**2)
    ss_tot = np.sum((dc4 - dc4_mean)**2)

    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


for pname, patch, dc4_list, dir_r2s, lap_r2s, j05, j15, gauge in patches_data:
    print(f"--- {pname} ---")

    # Compute graph distances
    gdist = compute_graph_distances(patch['qubits'], patch['edges'], patch['butterfly_phys'])
    ptol = phys_to_local(patch)

    # Distance for each local index
    distances_local = [0] * 9
    for phys_q, local_i in ptol.items():
        distances_local[local_i] = gdist[phys_q]

    print(f"  Qubit distances from butterfly: {distances_local}")
    print(f"  Qubit list: {patch['qubits']}")
    print(f"  j=1/2 local indices: {j05} (dist: {[distances_local[i] for i in j05]})")
    print(f"  j=3/2 local indices: {j15} (dist: {[distances_local[i] for i in j15]})")
    print(f"  gauge local indices: {gauge} (dist: {[distances_local[i] for i in gauge]})")

    # Compute mean DC4 at each distance
    mean_dc4_all = np.mean(dc4_list, axis=0)
    unique_dists = sorted(set(distances_local))
    print(f"\n  Mean DC4 by distance from butterfly:")
    for d in unique_dists:
        qubits_at_d = [i for i, dd in enumerate(distances_local) if dd == d]
        mean_val = np.mean([mean_dc4_all[i] for i in qubits_at_d])
        print(f"    d={d}: qubits {qubits_at_d}, mean DC4 = {mean_val:.4f}")

    # Per-seed distance R² and Dirac R²
    dist_r2s = []
    dirac_r2s_recomputed = []
    for s, dc4 in enumerate(dc4_list):
        dr2 = distance_model_r2(dc4, distances_local)
        dist_r2s.append(dr2)
        dirac_r2 = compute_dirac_r2(dc4, j05, j15, gauge)
        dirac_r2s_recomputed.append(dirac_r2)

    dist_r2s = np.array(dist_r2s)
    dirac_r2s_recomputed = np.array(dirac_r2s_recomputed)

    print(f"\n  Per-seed comparison:")
    print(f"  {'Seed':<6} {'Dirac R²':<12} {'Distance R²':<14} {'Dirac wins?'}")
    for s in range(len(dc4_list)):
        winner = "YES" if dirac_r2s_recomputed[s] > dist_r2s[s] else "no"
        print(f"  {s:<6} {dirac_r2s_recomputed[s]:<12.4f} {dist_r2s[s]:<14.4f} {winner}")

    print(f"\n  Mean Dirac R²:    {np.mean(dirac_r2s_recomputed):.4f} ± {np.std(dirac_r2s_recomputed):.4f}")
    print(f"  Mean Distance R²: {np.mean(dist_r2s):.4f} ± {np.std(dist_r2s):.4f}")

    # Mean-level fit
    mean_dist_r2 = distance_model_r2(mean_dc4_all, distances_local)
    mean_dirac_r2 = compute_dirac_r2(mean_dc4_all, j05, j15, gauge)
    print(f"\n  On mean DC4 vector:")
    print(f"    Dirac R²:    {mean_dirac_r2:.4f}")
    print(f"    Distance R²: {mean_dist_r2:.4f}")

    dirac_wins = np.sum(dirac_r2s_recomputed > dist_r2s)
    print(f"\n  Dirac beats distance model in {dirac_wins}/{len(dc4_list)} seeds")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: EFFECT SIZE AND CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("ANALYSIS 3: EFFECT SIZE, BOOTSTRAP CI, PAIRED t-TEST")
print("=" * 72)
print("Tests the magnitude and reliability of the Dirac advantage over Laplacian.\n")

N_BOOT = 10000

def cohens_d(x, y):
    """Cohen's d for paired samples (x - y)."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def bootstrap_ci(x, y, n_boot=N_BOOT, ci=0.95):
    """Bootstrap 95% CI for mean(x - y)."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    rng = np.random.RandomState(999)

    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return lo, hi, boot_means

all_patches_for_analysis3 = [
    ('Patch A', a_dir_r2s, a_lap_r2s, 10),
    ('Patch B', b_dir_r2s, b_lap_r2s, 10),
    ('Patch C', c_dir_r2s, c_lap_r2s, 5),
]

for pname, dir_r2s, lap_r2s, n_seeds in all_patches_for_analysis3:
    print(f"--- {pname} ({n_seeds} seeds) ---")
    diff = dir_r2s - lap_r2s
    mean_diff = np.mean(diff)
    print(f"  Mean Dirac R²:    {np.mean(dir_r2s):.4f} ± {np.std(dir_r2s, ddof=1):.4f}")
    print(f"  Mean Laplacian R²:{np.mean(lap_r2s):.4f} ± {np.std(lap_r2s, ddof=1):.4f}")
    print(f"  Mean difference:  {mean_diff:.4f}")

    # Cohen's d
    d = cohens_d(dir_r2s, lap_r2s)
    if abs(d) < 0.2:
        effect_label = "negligible"
    elif abs(d) < 0.5:
        effect_label = "small"
    elif abs(d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"
    print(f"  Cohen's d:        {d:.3f} ({effect_label})")

    # Bootstrap CI
    lo, hi, boot_dist = bootstrap_ci(dir_r2s, lap_r2s)
    print(f"  95% Bootstrap CI: [{lo:.4f}, {hi:.4f}]")
    ci_excludes_zero = "YES" if lo > 0 else "no"
    print(f"  CI excludes zero: {ci_excludes_zero}")

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(dir_r2s, lap_r2s)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  Paired t-test:    t = {t_stat:.3f}, p = {p_val:.6f} {sig}")

    # Per-seed breakdown
    print(f"\n  Per-seed R² values:")
    print(f"  {'Seed':<6} {'Dirac R²':<12} {'Laplacian R²':<14} {'Diff':<10} {'Dirac wins?'}")
    for s in range(n_seeds):
        w = "YES" if dir_r2s[s] > lap_r2s[s] else "no"
        print(f"  {s:<6} {dir_r2s[s]:<12.4f} {lap_r2s[s]:<14.4f} {diff[s]:<10.4f} {w}")

    wins = np.sum(dir_r2s > lap_r2s)
    print(f"\n  Dirac wins {wins}/{n_seeds} seeds ({100*wins/n_seeds:.0f}%)")

    # Sign test (non-parametric)
    sign_p = stats.binom_test(wins, n_seeds, 0.5) if hasattr(stats, 'binom_test') else \
             stats.binomtest(wins, n_seeds, 0.5).pvalue
    print(f"  Sign test p-value: {sign_p:.6f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED ANALYSIS ACROSS ALL PATCHES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("COMBINED ANALYSIS (ALL 25 SEEDS ACROSS 3 PATCHES)")
print("=" * 72)

all_dir = np.concatenate([a_dir_r2s, b_dir_r2s, c_dir_r2s])
all_lap = np.concatenate([a_lap_r2s, b_lap_r2s, c_lap_r2s])
all_diff = all_dir - all_lap

print(f"Total seeds: {len(all_dir)}")
print(f"Mean Dirac R²:    {np.mean(all_dir):.4f} ± {np.std(all_dir, ddof=1):.4f}")
print(f"Mean Laplacian R²:{np.mean(all_lap):.4f} ± {np.std(all_lap, ddof=1):.4f}")
print(f"Mean difference:  {np.mean(all_diff):.4f}")

d_all = cohens_d(all_dir, all_lap)
if abs(d_all) < 0.2:
    elabel = "negligible"
elif abs(d_all) < 0.5:
    elabel = "small"
elif abs(d_all) < 0.8:
    elabel = "medium"
else:
    elabel = "large"
print(f"Cohen's d:        {d_all:.3f} ({elabel})")

lo_all, hi_all, _ = bootstrap_ci(all_dir, all_lap)
print(f"95% Bootstrap CI: [{lo_all:.4f}, {hi_all:.4f}]")
print(f"CI excludes zero: {'YES' if lo_all > 0 else 'no'}")

t_all, p_all = stats.ttest_rel(all_dir, all_lap)
sig_all = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else "ns"
print(f"Paired t-test:    t = {t_all:.3f}, p = {p_all:.8f} {sig_all}")

wins_all = np.sum(all_dir > all_lap)
print(f"Dirac wins:       {wins_all}/{len(all_dir)} seeds ({100*wins_all/len(all_dir):.0f}%)")

sign_p_all = stats.binomtest(int(wins_all), len(all_dir), 0.5).pvalue
print(f"Sign test:        p = {sign_p_all:.8f}")

# Wilcoxon signed-rank (non-parametric alternative to paired t)
w_stat, w_p = stats.wilcoxon(all_dir, all_lap)
print(f"Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_p:.8f}")


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("VERDICT SUMMARY")
print("=" * 72)

print("""
Key questions for publishability:

1. PERMUTATION TEST: Is the Dirac qubit grouping special?
   → If combined p < 0.05, the specific j=1/2, j=3/2, gauge assignment
     captures more variance than random 2+4+3 splits.

2. DISTANCE MODEL: Is it just radial decay?
   → If Dirac R² substantially exceeds distance R², the mode structure
     goes beyond simple proximity to the butterfly qubit.

3. EFFECT SIZE: Is the Dirac advantage meaningful?
   → Cohen's d > 0.8 = large effect; CI excluding zero = reliable.
   → p < 0.05 across multiple tests = statistically significant.
""")
