"""
curvature_analysis.py
=====================
Computes Gaussian curvature for:
  1. Each individual aligned sheet (per group)
  2. Each group's mean sheet
  3. Each group's simulated plate
  4. Differences in curvature between mean sheet and simulated plate
  5. Cross-group comparisons of curvature

Run AFTER align_and_compare.py, which produces:
  - <OUTPUT_DIR>/<group>/<sheet>_aligned.ply
  - <OUTPUT_DIR>/<group>/<group>_mean_sheet.ply

Dependencies:
    pip install plyfile numpy scipy matplotlib

Usage:
    python curvature_analysis.py
"""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from plyfile import PlyData, PlyElement
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\felic\Deviation\Scans_trim"
SIM_DIR    = r"C:\Users\felic\Deviation\Sim_noedges"
OUTPUT_DIR = r"C:\Users\felic\Deviation\Analys"

GROUPS = ["A_1", "A_1.5", "B_1", "B_1.5", "C_1", "C_1.5"]

COMPARE_PAIRS = [
    ("A_1", "A_1.5"),
    ("B_1", "B_1.5"),
    ("C_1", "C_1.5"),
]

# Mapping from group name to simulated PLY file name in SIM_DIR.
# Must match the names used in align_and_compare.py
SIM_MAP = {
    "A_1":   "A_1000_trimmed.ply",
    "A_1.5": "A_1000_trimmed.ply",
    "B_1":   "B_1000_trimmed.ply",
    "B_1.5": "B_1000_trimmed.ply",
    "C_1":   "C_1000_trimmed.ply",
    "C_1.5": "C_1000_trimmed.ply",
}

CORNER_REGION_K = 200
GRID_SPACING    = 1.0  # mm

# Gaussian curvature: sigma for smoothing before differentiation
# (in grid cells)
CURVATURE_SIGMA = 3.0


# ──────────────────────────────────────────────
#  PLY I/O
# ──────────────────────────────────────────────
def load_ply(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]])


def save_ply(pts, out_path, scalars=None):
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if scalars:
        for name in scalars:
            fields.append((name, "f4"))
    arr = np.empty(len(pts), dtype=fields)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    if scalars:
        for name, values in scalars.items():
            arr[name] = values
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=True).write(str(out_path))


# ──────────────────────────────────────────────
#  Landmarks and Kabsch (needed for simulated plate alignment)
# ──────────────────────────────────────────────
def find_landmarks(pts, k=CORNER_REGION_K):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    u, v = eigvecs[:, 2], eigvecs[:, 1]
    pts_2d = np.column_stack([centered @ u, centered @ v])

    hull = ConvexHull(pts_2d)
    hull_pts = pts_2d[hull.vertices]
    best_area = np.inf
    n_hull = len(hull_pts)
    for i in range(n_hull):
        edge = hull_pts[(i + 1) % n_hull] - hull_pts[i]
        angle = np.arctan2(edge[1], edge[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = pts_2d @ rot.T
        x_min, x_max = rotated[:, 0].min(), rotated[:, 0].max()
        y_min, y_max = rotated[:, 1].min(), rotated[:, 1].max()
        if (x_max - x_min) * (y_max - y_min) < best_area:
            best_area = (x_max - x_min) * (y_max - y_min)
            best_bounds = (x_min, x_max, y_min, y_max)
            best_rot = rot

    x_min, x_max, y_min, y_max = best_bounds
    corners_rot = np.array([[x_min, y_min], [x_max, y_min],
                             [x_max, y_max], [x_min, y_max]])
    corners_2d = corners_rot @ best_rot
    tree_2d = cKDTree(pts_2d)
    corners_3d = np.zeros((4, 3))
    for i, c2d in enumerate(corners_2d):
        _, idx = tree_2d.query(c2d, k=k)
        corners_3d[i] = pts[idx].mean(axis=0)

    corner_centroid = corners_3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(corners_3d - corner_centroid)
    plane_normal = Vt[-1]
    abs_dists = np.abs((pts - corner_centroid) @ plane_normal)
    crease_vertex = pts[abs_dists >= np.percentile(abs_dists, 99)].mean(axis=0)

    return np.vstack([corners_3d, crease_vertex.reshape(1, 3)])


def kabsch_align(src, tgt):
    sc, tc = src.mean(0), tgt.mean(0)
    H = (src - sc).T @ (tgt - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    return R, tc - R @ sc


def apply_transform(pts, R, t):
    return (R @ pts.T).T + t


def compute_rmse(a, b):
    return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))


def find_best_landmark_alignment(src_lm, tgt_lm):
    sc, sp = src_lm[:4], src_lm[4:5]
    tc, tp = tgt_lm[:4], tgt_lm[4:5]
    best_rmse, best_R, best_t, best_rot = np.inf, None, None, 0
    for flip in [False, True]:
        corners = sc[::-1] if flip else sc
        for rot in range(4):
            s_all = np.vstack([np.roll(corners, rot, axis=0), sp])
            t_all = np.vstack([tc, tp])
            R, t = kabsch_align(s_all, t_all)
            rmse = compute_rmse(apply_transform(s_all, R, t), t_all)
            if rmse < best_rmse:
                best_rmse, best_R, best_t = rmse, R, t
                best_rot = rot + (4 if flip else 0)
    return best_R, best_t, best_rmse, best_rot


# ──────────────────────────────────────────────
#  Gaussian curvature from height field z(x,y)
# ──────────────────────────────────────────────
def compute_gaussian_curvature(z_grid, spacing, sigma=CURVATURE_SIGMA):
    """
    K = (z_xx * z_yy - z_xy^2) / (1 + z_x^2 + z_y^2)^2

    Smooths z with Gaussian filter before differentiation.
    Returns K array (same shape), NaN where input is NaN.
    """
    nan_mask = np.isnan(z_grid)
    z = z_grid.copy()
    z[nan_mask] = 0.0

    z_smooth = gaussian_filter(z, sigma=sigma)
    z_smooth[nan_mask] = np.nan

    z_y, z_x = np.gradient(z_smooth, spacing)
    z_xx = np.gradient(np.gradient(z_smooth, spacing, axis=1),
                        spacing, axis=1)
    z_yy = np.gradient(np.gradient(z_smooth, spacing, axis=0),
                        spacing, axis=0)
    z_xy = np.gradient(np.gradient(z_smooth, spacing, axis=1),
                        spacing, axis=0)

    denom = (1.0 + z_x**2 + z_y**2)**2
    K = (z_xx * z_yy - z_xy**2) / denom
    K[nan_mask] = np.nan
    return K


def curvature_stats(K_valid):
    """Summary stats for a Gaussian curvature array (NaN-free)."""
    return {
        "K_mean":     float(np.mean(K_valid)),
        "K_std":      float(np.std(K_valid)),
        "K_min":      float(np.min(K_valid)),
        "K_max":      float(np.max(K_valid)),
        "K_abs_mean": float(np.mean(np.abs(K_valid))),
        "K_abs_max":  float(np.max(np.abs(K_valid))),
    }


def save_curvature_map(gx, gy, z, K, valid, name, out_dir):
    """Save Gaussian curvature as PLY, CSV, and PNG."""
    pts    = np.column_stack([gx[valid], gy[valid], z[valid]])
    K_vals = K[valid]

    # PLY
    save_ply(pts, out_dir / f"{name}_gaussian_K.ply",
             scalars={"gaussian_curvature": K_vals})

    # CSV
    csv_path = out_dir / f"{name}_gaussian_K.csv"
    np.savetxt(csv_path,
               np.column_stack([pts, K_vals]),
               delimiter=",", header="x,y,z,gaussian_curvature",
               comments="", fmt="%.6f")

    # PNG
    fig, ax = plt.subplots(figsize=(10, 7))
    K_clean = K_vals[~np.isnan(K_vals)]
    vmax = np.percentile(np.abs(K_clean), 98) if len(K_clean) > 0 else 1
    sc = ax.scatter(gx[valid], gy[valid], c=K_vals,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=0.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Gaussian curvature K (1/mm²)")
    ax.set_aspect("equal")
    ax.set_title(f"{name} - Gaussian curvature")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_gaussian_K.png", dpi=200)
    plt.close("all")

    return curvature_stats(K_clean)


# ──────────────────────────────────────────────
#  Analyse curvature for one group
# ──────────────────────────────────────────────
def analyse_group_curvature(group_prefix, in_dir, sim_dir, out_dir):
    """
    Compute Gaussian curvature for:
      - each aligned sheet
      - the mean sheet
      - the simulated plate (after aligning to group reference)
    Also compute K differences between sheet/mean and simulated plate.
    """
    group_dir = out_dir / group_prefix
    group_dir.mkdir(parents=True, exist_ok=True)
    curv_dir = group_dir / "curvature"
    curv_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CURVATURE ANALYSIS - GROUP {group_prefix}")
    print(f"{'='*60}")

    # 1. Load aligned sheets produced by align_and_compare.py
    aligned_files = sorted(group_dir.glob("*_aligned.ply"))
    if not aligned_files:
        print(f"  No aligned PLY files found in {group_dir}")
        print(f"  Run align_and_compare.py first")
        return None

    aligned = []
    for f in aligned_files:
        name = f.stem.replace("_aligned", "")
        pts = load_ply(f)
        aligned.append({"name": name, "pts": pts})
        print(f"  Loaded {name}: {len(pts)} pts")

    # 2. Build common grid from aligned sheets
    x_lo = max(a["pts"][:, 0].min() for a in aligned) + GRID_SPACING
    x_hi = min(a["pts"][:, 0].max() for a in aligned) - GRID_SPACING
    y_lo = max(a["pts"][:, 1].min() for a in aligned) + GRID_SPACING
    y_hi = min(a["pts"][:, 1].max() for a in aligned) - GRID_SPACING
    gx, gy = np.meshgrid(np.arange(x_lo, x_hi, GRID_SPACING),
                          np.arange(y_lo, y_hi, GRID_SPACING))

    z_stack = np.full((len(aligned), *gx.shape), np.nan)
    for i, a in enumerate(aligned):
        z_stack[i] = griddata(a["pts"][:, :2], a["pts"][:, 2],
                                (gx, gy), method="linear")

    with np.errstate(all="ignore"):
        z_mean = np.nanmean(z_stack, axis=0)
    valid_mean = ~np.isnan(z_mean)

    # 3. Per-sheet Gaussian curvature
    sheet_K_stats = {}
    summary_csv = curv_dir / f"{group_prefix}_curvature_summary.csv"
    with open(summary_csv, "w") as f:
        f.write("name,K_mean,K_std,K_min,K_max,K_abs_mean,K_abs_max\n")

        for i, a in enumerate(aligned):
            valid = ~np.isnan(z_stack[i])
            K = compute_gaussian_curvature(z_stack[i], GRID_SPACING)
            ks = save_curvature_map(gx, gy, z_stack[i], K, valid,
                                     a["name"], curv_dir)
            sheet_K_stats[a["name"]] = ks
            f.write(f"{a['name']},{ks['K_mean']:.8f},{ks['K_std']:.8f},"
                     f"{ks['K_min']:.8f},{ks['K_max']:.8f},"
                     f"{ks['K_abs_mean']:.8f},{ks['K_abs_max']:.8f}\n")
            print(f"    {a['name']}: K mean={ks['K_mean']:.6f}, "
                  f"|K| mean={ks['K_abs_mean']:.6f}")

        # Mean sheet curvature
        K_mean = compute_gaussian_curvature(z_mean, GRID_SPACING)
        K_mean_stats = save_curvature_map(gx, gy, z_mean, K_mean, valid_mean,
                                           f"{group_prefix}_mean_sheet",
                                           curv_dir)
        f.write(f"{group_prefix}_mean,{K_mean_stats['K_mean']:.8f},"
                f"{K_mean_stats['K_std']:.8f},"
                f"{K_mean_stats['K_min']:.8f},"
                f"{K_mean_stats['K_max']:.8f},"
                f"{K_mean_stats['K_abs_mean']:.8f},"
                f"{K_mean_stats['K_abs_max']:.8f}\n")
        print(f"\n    Mean sheet: K mean={K_mean_stats['K_mean']:.6f}, "
              f"|K| mean={K_mean_stats['K_abs_mean']:.6f}")

    # 4. Simulated plate: align, grid, curvature
    sim_file = SIM_MAP.get(group_prefix)
    sim_result = None
    if sim_file is None:
        print(f"\n  No simulated plate mapping for {group_prefix}")
    else:
        sim_path = sim_dir / sim_file
        if not sim_path.exists():
            print(f"\n  Simulated plate not found: {sim_path}")
        else:
            print(f"\n  Loading simulated plate: {sim_file}")
            pts_sim_raw = load_ply(sim_path)
            lm_sim      = find_landmarks(pts_sim_raw)

            # Align simulated plate to reference (first aligned sheet)
            ref_pts = aligned[0]["pts"]
            ref_lm  = find_landmarks(ref_pts)
            R, t, rmse_sim, _ = find_best_landmark_alignment(lm_sim, ref_lm)
            pts_sim = apply_transform(pts_sim_raw, R, t)
            print(f"  Simulated plate alignment RMSE: {rmse_sim:.2f} mm")

            # Grid simulated plate
            z_sim = griddata(pts_sim[:, :2], pts_sim[:, 2],
                              (gx, gy), method="linear")
            valid_sim = ~np.isnan(z_sim)

            # Compute simulated curvature
            K_sim = compute_gaussian_curvature(z_sim, GRID_SPACING)
            K_sim_stats = save_curvature_map(
                gx, gy, z_sim, K_sim, valid_sim,
                f"{group_prefix}_simulated", curv_dir)
            print(f"  Simulated K mean={K_sim_stats['K_mean']:.6f}, "
                  f"|K| mean={K_sim_stats['K_abs_mean']:.6f}")

            # K difference: mean sheet vs simulated
            valid_both = valid_mean & valid_sim
            K_diff_mean = K_mean - K_sim
            Kd_mean_valid = K_diff_mean[valid_both]
            Kd_mean_clean = Kd_mean_valid[~np.isnan(Kd_mean_valid)]

            if len(Kd_mean_clean) > 0:
                save_ply(np.column_stack([gx[valid_both], gy[valid_both],
                                            z_mean[valid_both]]),
                         curv_dir / f"{group_prefix}_mean_vs_sim_K_diff.ply",
                         scalars={"K_diff": Kd_mean_valid})

                with open(curv_dir /
                          f"{group_prefix}_mean_vs_sim_K_diff_summary.csv",
                          "w") as f:
                    f.write("metric,value\n")
                    f.write(f"K_diff_mean,{np.mean(Kd_mean_clean):.8f}\n"
                            f"K_diff_std,{np.std(Kd_mean_clean):.8f}\n"
                            f"K_diff_abs_mean,"
                            f"{np.mean(np.abs(Kd_mean_clean)):.8f}\n"
                            f"K_diff_abs_max,"
                            f"{np.max(np.abs(Kd_mean_clean)):.8f}\n")

                # Plot K diff
                fig, ax = plt.subplots(figsize=(10, 7))
                vmax_K = np.percentile(np.abs(Kd_mean_clean), 98)
                sc = ax.scatter(gx[valid_both], gy[valid_both],
                                 c=Kd_mean_valid, cmap="RdBu_r",
                                 vmin=-vmax_K, vmax=vmax_K, s=.5,
                                 edgecolors="none")
                fig.colorbar(sc, ax=ax, label="K diff (1/mm²)")
                ax.set_aspect("equal")
                ax.set_title(f"{group_prefix} mean vs sim - "
                              f"Gaussian K difference")
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                plt.tight_layout()
                plt.savefig(curv_dir /
                             f"{group_prefix}_mean_vs_sim_K_diff.png",
                             dpi=200)
                plt.close("all")
                print(f"  Mean vs sim K diff: "
                      f"mean={np.mean(Kd_mean_clean):.6f}, "
                      f"|K diff| mean={np.mean(np.abs(Kd_mean_clean)):.6f}")

            # K differences per sheet vs simulated
            sheet_K_diff_stats = {}
            sheet_csv = curv_dir / f"{group_prefix}_sheets_vs_sim_K_diff.csv"
            with open(sheet_csv, "w") as f:
                f.write("name,K_diff_mean,K_diff_std,"
                        "K_diff_abs_mean,K_diff_abs_max\n")
                for i, a in enumerate(aligned):
                    K_sheet = compute_gaussian_curvature(z_stack[i],
                                                          GRID_SPACING)
                    valid_s = (~np.isnan(z_stack[i])) & valid_sim
                    Kd_s = (K_sheet - K_sim)[valid_s]
                    Kd_sc = Kd_s[~np.isnan(Kd_s)]
                    if len(Kd_sc) == 0:
                        continue
                    st = {
                        "K_diff_mean":     float(np.mean(Kd_sc)),
                        "K_diff_std":      float(np.std(Kd_sc)),
                        "K_diff_abs_mean": float(np.mean(np.abs(Kd_sc))),
                        "K_diff_abs_max":  float(np.max(np.abs(Kd_sc))),
                    }
                    sheet_K_diff_stats[a["name"]] = st
                    f.write(f"{a['name']},{st['K_diff_mean']:.8f},"
                             f"{st['K_diff_std']:.8f},"
                             f"{st['K_diff_abs_mean']:.8f},"
                             f"{st['K_diff_abs_max']:.8f}\n")

            sim_result = {
                "K_sim_stats":        K_sim_stats,
                "K_diff_mean_stats": {
                    "K_diff_mean":     float(np.mean(Kd_mean_clean)),
                    "K_diff_std":      float(np.std(Kd_mean_clean)),
                    "K_diff_abs_mean": float(np.mean(np.abs(Kd_mean_clean))),
                    "K_diff_abs_max":  float(np.max(np.abs(Kd_mean_clean))),
                } if len(Kd_mean_clean) > 0 else None,
                "sheet_K_diff_stats": sheet_K_diff_stats,
                "alignment_rmse":     rmse_sim,
            }

    return {
        "group": group_prefix, "gx": gx, "gy": gy,
        "z_mean": z_mean, "K_mean": K_mean, "valid": valid_mean,
        "sheet_K_stats": sheet_K_stats,
        "K_mean_stats":  K_mean_stats,
        "sim_result":    sim_result,
    }


# ──────────────────────────────────────────────
#  Compare curvature between two groups (mean sheets)
# ──────────────────────────────────────────────
def compare_curvature_between_groups(data_a, data_b, out_dir):
    na, nb = data_a["group"], data_b["group"]
    print(f"\n{'='*60}\n  CURVATURE COMPARISON: {na} vs {nb}\n{'='*60}")

    comp_dir = out_dir / f"{na}_vs_{nb}" / "curvature"
    comp_dir.mkdir(parents=True, exist_ok=True)

    gx_a, gy_a = data_a["gx"], data_a["gy"]
    gx_b, gy_b = data_b["gx"], data_b["gy"]

    x_lo = max(gx_a.min(), gx_b.min()) + GRID_SPACING
    x_hi = min(gx_a.max(), gx_b.max()) - GRID_SPACING
    y_lo = max(gy_a.min(), gy_b.min()) + GRID_SPACING
    y_hi = min(gy_a.max(), gy_b.max()) - GRID_SPACING
    gx, gy = np.meshgrid(np.arange(x_lo, x_hi, GRID_SPACING),
                          np.arange(y_lo, y_hi, GRID_SPACING))

    pts_a = np.column_stack([gx_a[data_a["valid"]],
                              gy_a[data_a["valid"]],
                              data_a["K_mean"][data_a["valid"]]])
    pts_b = np.column_stack([gx_b[data_b["valid"]],
                              gy_b[data_b["valid"]],
                              data_b["K_mean"][data_b["valid"]]])

    K_a = griddata(pts_a[:, :2], pts_a[:, 2], (gx, gy), method="linear")
    K_b = griddata(pts_b[:, :2], pts_b[:, 2], (gx, gy), method="linear")
    K_diff = K_b - K_a
    valid = ~np.isnan(K_diff)
    Kd = K_diff[valid]

    stats = {
        "K_diff_mean":     float(np.mean(Kd)),
        "K_diff_std":      float(np.std(Kd)),
        "K_diff_abs_mean": float(np.mean(np.abs(Kd))),
        "K_diff_abs_max":  float(np.max(np.abs(Kd))),
    }
    print(f"  K diff ({nb}-{na}): mean={stats['K_diff_mean']:.6f}, "
          f"|K diff| mean={stats['K_diff_abs_mean']:.6f}")

    # Save outputs
    pts_c = np.column_stack([gx[valid], gy[valid],
                              .5 * (K_a[valid] + K_b[valid])])
    save_ply(pts_c, comp_dir / f"{na}_vs_{nb}_K_diff.ply",
             scalars={"K_diff": Kd})

    with open(comp_dir / f"{na}_vs_{nb}_K_diff_summary.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in stats.items():
            f.write(f"{k},{v:.8f}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    vmax_K = np.percentile(np.abs(Kd), 98)
    sc = ax.scatter(gx[valid], gy[valid], c=Kd, cmap="RdBu_r",
                     vmin=-vmax_K, vmax=vmax_K, s=.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="K diff (1/mm²)")
    ax.set_aspect("equal")
    ax.set_title(f"Gaussian K: {nb} - {na}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(comp_dir / f"{na}_vs_{nb}_K_diff.png", dpi=200)
    plt.close("all")

    return stats


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    in_dir  = Path(INPUT_DIR)
    sim_dir = Path(SIM_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Per-group curvature analysis
    group_data = {}
    for g in GROUPS:
        data = analyse_group_curvature(g, in_dir, sim_dir, out_dir)
        if data:
            group_data[g] = data

    # 2. Cross-group curvature comparison
    cross_comp = {}
    for a, b in COMPARE_PAIRS:
        if a in group_data and b in group_data:
            cross_comp[(a, b)] = compare_curvature_between_groups(
                group_data[a], group_data[b], out_dir)

    # 3. Master CSV
    master = out_dir / "master_curvature_results.csv"
    with open(master, "w") as f:
        f.write("=== MEAN SHEET CURVATURE ===\n")
        f.write("group,K_mean,K_std,K_min,K_max,K_abs_mean,K_abs_max\n")
        for g in sorted(group_data):
            ks = group_data[g]["K_mean_stats"]
            f.write(f"{g},{ks['K_mean']:.8f},{ks['K_std']:.8f},"
                    f"{ks['K_min']:.8f},{ks['K_max']:.8f},"
                    f"{ks['K_abs_mean']:.8f},{ks['K_abs_max']:.8f}\n")

        f.write("\n=== SHEET CURVATURE (all sheets) ===\n")
        f.write("group,sheet,K_mean,K_std,K_min,K_max,K_abs_mean,K_abs_max\n")
        for g in sorted(group_data):
            for sheet, ks in group_data[g]["sheet_K_stats"].items():
                f.write(f"{g},{sheet},{ks['K_mean']:.8f},{ks['K_std']:.8f},"
                        f"{ks['K_min']:.8f},{ks['K_max']:.8f},"
                        f"{ks['K_abs_mean']:.8f},{ks['K_abs_max']:.8f}\n")

        f.write("\n=== SIMULATED PLATE CURVATURE ===\n")
        f.write("group,K_mean,K_std,K_min,K_max,K_abs_mean,K_abs_max,"
                "alignment_rmse\n")
        for g in sorted(group_data):
            sr = group_data[g].get("sim_result")
            if not sr:
                continue
            ks = sr["K_sim_stats"]
            f.write(f"{g},{ks['K_mean']:.8f},{ks['K_std']:.8f},"
                    f"{ks['K_min']:.8f},{ks['K_max']:.8f},"
                    f"{ks['K_abs_mean']:.8f},{ks['K_abs_max']:.8f},"
                    f"{sr['alignment_rmse']:.4f}\n")

        f.write("\n=== MEAN SHEET vs SIMULATED PLATE (K difference) ===\n")
        f.write("group,K_diff_mean,K_diff_std,K_diff_abs_mean,K_diff_abs_max\n")
        for g in sorted(group_data):
            sr = group_data[g].get("sim_result")
            if not sr or not sr.get("K_diff_mean_stats"):
                continue
            st = sr["K_diff_mean_stats"]
            f.write(f"{g},{st['K_diff_mean']:.8f},{st['K_diff_std']:.8f},"
                    f"{st['K_diff_abs_mean']:.8f},"
                    f"{st['K_diff_abs_max']:.8f}\n")

        f.write("\n=== EACH SHEET vs SIMULATED (K difference) ===\n")
        f.write("group,sheet,K_diff_mean,K_diff_std,"
                "K_diff_abs_mean,K_diff_abs_max\n")
        for g in sorted(group_data):
            sr = group_data[g].get("sim_result")
            if not sr:
                continue
            for sheet, st in sr["sheet_K_diff_stats"].items():
                f.write(f"{g},{sheet},{st['K_diff_mean']:.8f},"
                        f"{st['K_diff_std']:.8f},"
                        f"{st['K_diff_abs_mean']:.8f},"
                        f"{st['K_diff_abs_max']:.8f}\n")

        f.write("\n=== CROSS-GROUP K DIFFERENCE (mean sheets) ===\n")
        f.write("comparison,K_diff_mean,K_diff_std,"
                "K_diff_abs_mean,K_diff_abs_max\n")
        for (a, b), st in sorted(cross_comp.items()):
            f.write(f"{a}_vs_{b},{st['K_diff_mean']:.8f},"
                    f"{st['K_diff_std']:.8f},"
                    f"{st['K_diff_abs_mean']:.8f},"
                    f"{st['K_diff_abs_max']:.8f}\n")

    print(f"\n{'='*60}")
    print(f"  MASTER CSV: {master}")
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"  Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
