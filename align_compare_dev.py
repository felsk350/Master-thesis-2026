"""
align_and_compare.py
====================
Aligns sheets within each group, computes mean sheets,
and performs deviation analysis (no Gaussian curvature - see
curvature_analysis.py for that).

Per-group:
  1. Find 5 landmarks (4 corners + crease vertex)
  2. Kabsch alignment to reference sheet
  3. Interpolate to grid -> mean sheet + std
  4. Deviation per sheet relative to mean sheet
  5. Identify sheet closest to mean

Cross-group:
  6. Compare mean sheets
  7. Compare closest-to-mean sheets

Comparison against simulated plates (sim_point_noedges):
  8. Each sheet vs simulated plate (per group)
  9. Mean sheet vs simulated plate (per group)

Dependencies:
    pip install plyfile numpy scipy matplotlib

Usage:
    python align_and_compare.py
"""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import griddata
from plyfile import PlyData, PlyElement
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
INPUT_DIR = r"\\stuur02.it.liu.se\students\lovso390\Downloads\Kod\scan_noedges"
SIM_DIR  = r"\\stuur02.it.liu.se\students\lovso390\Downloads\Kod\sim_rep"
OUTPUT_DIR  = r"./deviation_output"

GROUPS = ["A_1", "A_1.5", "B_1", "B_1.5", "C_1", "C_1.5"]

COMPARE_PAIRS = [
    ("A_1", "A_1.5"),
    ("B_1", "B_1.5"),
    ("C_1", "C_1.5"),
]

# Mapping from group name to simulated PLY file name in SIM_DIR.
# Adjust these file names to match the actual files in your Sim_noedges folder.
SIM_MAP = {
    "A_1":   "A_3_noedge.ply",
    "A_1.5": "A_3_noedge.ply",
    "B_1":   "B_5_noedge.ply",
    "B_1.5": "B_5_noedge.ply",
    "C_1":   "C_5_noedge.ply",
    "C_1.5": "C_5_noedge.ply",
}

CORNER_REGION_K = 200
GRID_SPACING    = 1.0  # mm

# Reference scan for each group (all others align to this one)
GROUP_REFS = {
    "A_1":   "A_1_4",
    "A_1.5": "A_1.5_1",
    "B_1":   "B_1_1",
    "B_1.5": "B_1.5_3",
    "C_1":   "C_1_1",
    "C_1.5": "C_1.5_5",
}


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
#  Landmarks (4 corners + crease vertex)
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

    # Crease vertex: centroid of points furthest from the corner plane
    corner_centroid = corners_3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(corners_3d - corner_centroid)
    plane_normal = Vt[-1]
    abs_dists = np.abs((pts - corner_centroid) @ plane_normal)
    crease_vertex = pts[abs_dists >= np.percentile(abs_dists, 99)].mean(axis=0)

    return np.vstack([corners_3d, crease_vertex.reshape(1, 3)])


# ──────────────────────────────────────────────
#  Kabsch alignment
# ──────────────────────────────────────────────
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
#  Deviation stats
# ──────────────────────────────────────────────
def deviation_stats(d):
    n_pos, n_neg, n_tot = np.sum(d > 0), np.sum(d < 0), len(d)
    return {
        "mean":    float(np.mean(d)),
        "std":     float(np.std(d)),
        "rms":     float(np.sqrt(np.mean(d**2))),
        "max_pos": float(np.max(d)) if n_pos > 0 else 0.0,
        "max_neg": float(np.min(d)) if n_neg > 0 else 0.0,
        "pct_pos": 100 * n_pos / n_tot if n_tot else 0.0,
        "pct_neg": 100 * n_neg / n_tot if n_tot else 0.0,
    }


# ──────────────────────────────────────────────
#  Process one group: alignment + mean sheet + deviations
# ──────────────────────────────────────────────
def process_group(group_prefix, in_dir, out_dir):
    group_dir = out_dir / group_prefix
    group_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(in_dir.glob(f"{group_prefix}_*_noedge.ply"))
    print(f"\n{'='*60}")
    print(f"  GROUP {group_prefix} ({len(ply_files)} files)")
    print(f"{'='*60}")

    if len(ply_files) < 2:
        print("  Need at least 2 files!")
        return None

    # Load and find landmarks
    sheets = []
    for f in ply_files:
        name = f.stem.replace("_noedge", "")
        print(f"  Loading {name} ... ", end="", flush=True)
        pts = load_ply(f)
        lm = find_landmarks(pts)
        sheets.append({"name": name, "pts": pts, "landmarks": lm})
        print(f"{len(pts)} pts, landmarks OK")

    # Align to the designated reference sheet for this group
    ref_name = GROUP_REFS.get(group_prefix)
    ref_matches = [s for s in sheets if s["name"] == ref_name]
    if not ref_matches:
        print(f"  WARNING: reference '{ref_name}' not found in group "
              f"{group_prefix}, falling back to first sheet.")
        ref = sheets[0]
    else:
        ref = ref_matches[0]
    ref_lm = ref["landmarks"]
    aligned = [{"name": ref["name"], "pts": ref["pts"], "corners": ref_lm[:4]}]
    save_ply(ref["pts"], group_dir / f"{ref['name']}_aligned.ply")
    print(f"\n  Reference: {ref['name']}")

    for s in sheets:
        if s["name"] == ref["name"]:
            continue
        R, t, rmse, rot_idx = find_best_landmark_alignment(s["landmarks"], ref_lm)
        pts_a = apply_transform(s["pts"], R, t)
        lm_a  = apply_transform(s["landmarks"], R, t)
        ce = np.linalg.norm(lm_a[:4] - ref_lm[:4], axis=1)
        ve = np.linalg.norm(lm_a[4] - ref_lm[4])
        flip = " (flipped)" if rot_idx >= 4 else ""
        print(f"  {s['name']}: RMSE={rmse:.2f}, "
              f"corners=[{', '.join(f'{e:.1f}' for e in ce)}], "
              f"vertex={ve:.1f} mm{flip}")
        save_ply(pts_a, group_dir / f"{s['name']}_aligned.ply")
        aligned.append({"name": s["name"], "pts": pts_a, "corners": lm_a[:4]})

    # Alignment check plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
              "#ff7f00", "#a65628", "#f781bf", "#999999"]
    rng = np.random.default_rng(42)
    for i, a in enumerate(aligned):
        c = colors[i % len(colors)]
        idx = rng.choice(len(a["pts"]), min(15000, len(a["pts"])), replace=False)
        p = a["pts"][idx]
        axes[0].scatter(p[:, 0], p[:, 1], s=.1, c=c, alpha=.3, label=a["name"])
        axes[1].scatter(p[:, 0], p[:, 2], s=.1, c=c, alpha=.3, label=a["name"])
        axes[2].scatter(p[:, 1], p[:, 2], s=.1, c=c, alpha=.3, label=a["name"])
    for ax, t, xl, yl in [(axes[0], "Top (XY)",  "X", "Y"),
                          (axes[1], "Side (XZ)", "X", "Z"),
                          (axes[2], "End (YZ)",  "Y", "Z")]:
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_xlabel(f"{xl} (mm)")
        ax.set_ylabel(f"{yl} (mm)")
        ax.legend(markerscale=10, fontsize=7)
    plt.suptitle(f"Group {group_prefix} - alignment", fontsize=14)
    plt.tight_layout()
    plt.savefig(group_dir / f"{group_prefix}_alignment.png", dpi=200)
    plt.close("all")

    # -- Mean sheet --
    print(f"\n  Computing mean sheet ...")
    x_lo = max(a["pts"][:, 0].min() for a in aligned) + GRID_SPACING
    x_hi = min(a["pts"][:, 0].max() for a in aligned) - GRID_SPACING
    y_lo = max(a["pts"][:, 1].min() for a in aligned) + GRID_SPACING
    y_hi = min(a["pts"][:, 1].max() for a in aligned) - GRID_SPACING
    gx, gy = np.meshgrid(np.arange(x_lo, x_hi, GRID_SPACING),
                          np.arange(y_lo, y_hi, GRID_SPACING))
    print(f"  Grid: {gx.shape[1]}x{gx.shape[0]} = {gx.size} points")

    z_stack = np.full((len(aligned), *gx.shape), np.nan)
    for i, a in enumerate(aligned):
        z_stack[i] = griddata(a["pts"][:, :2], a["pts"][:, 2],
                                (gx, gy), method="linear")

    with np.errstate(all="ignore"):
        z_mean = np.nanmean(z_stack, axis=0)
        z_std  = np.nanstd(z_stack, axis=0)

    valid    = ~np.isnan(z_mean)
    pts_mean = np.column_stack([gx[valid], gy[valid], z_mean[valid]])
    std_vals = z_std[valid]
    print(f"  Mean sheet: {len(pts_mean)} points, "
          f"Z-std: mean={std_vals.mean():.4f}, max={std_vals.max():.4f} mm")

    save_ply(pts_mean, group_dir / f"{group_prefix}_mean_sheet.ply")
    save_ply(pts_mean, group_dir / f"{group_prefix}_std_map.ply",
             scalars={"z_std_mm": std_vals})
    np.savetxt(group_dir / f"{group_prefix}_std_map.csv",
               np.column_stack([pts_mean, std_vals]),
               delimiter=",", header="x,y,z_mean,z_std_mm",
               comments="", fmt="%.4f")

    # -- Deviation per sheet --
    sheet_rms = {}
    summary_csv = group_dir / f"{group_prefix}_summary.csv"
    with open(summary_csv, "w") as f:
        f.write("name,mean_dev_mm,std_dev_mm,rms_dev_mm,"
                "max_pos_mm,max_neg_mm,pct_pos,pct_neg\n")
        for i, a in enumerate(aligned):
            diff = (z_stack[i] - z_mean)[valid]
            d    = diff[~np.isnan(diff)]
            st   = deviation_stats(d)
            sheet_rms[a["name"]] = st["rms"]

            save_ply(pts_mean, group_dir / f"{a['name']}_deviation.ply",
                     scalars={"deviation_mm": diff})
            sign    = np.where(diff >= 0, "positive", "negative")
            z_sheet = z_stack[i][valid]
            with open(group_dir / f"{a['name']}_deviation.csv", "w") as fc:
                fc.write("x,y,z_mean,z_sheet,deviation_mm,sign\n")
                for j in range(len(pts_mean)):
                    fc.write(f"{pts_mean[j, 0]:.4f},{pts_mean[j, 1]:.4f},"
                              f"{pts_mean[j, 2]:.4f},{z_sheet[j]:.4f},"
                              f"{diff[j]:.4f},{sign[j]}\n")

            f.write(f"{a['name']},{st['mean']:.4f},{st['std']:.4f},"
                     f"{st['rms']:.4f},{st['max_pos']:.4f},{st['max_neg']:.4f},"
                     f"{st['pct_pos']:.1f},{st['pct_neg']:.1f}\n")
            print(f"    {a['name']}: dev RMS={st['rms']:.3f}")

    closest_name = min(sheet_rms, key=sheet_rms.get)
    print(f"\n  * CLOSEST TO MEAN: {closest_name} "
          f"(RMS={sheet_rms[closest_name]:.4f} mm)")

    # Height + std maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(gx[valid], gy[valid], c=z_mean[valid],
                           cmap="coolwarm", s=.5, edgecolors="none")
    fig.colorbar(sc1, ax=axes[0], label="Z (mm)")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"{group_prefix} - mean sheet")
    sc2 = axes[1].scatter(gx[valid], gy[valid], c=z_std[valid],
                           cmap="hot", s=.5, edgecolors="none")
    fig.colorbar(sc2, ax=axes[1], label="Std (mm)")
    axes[1].set_aspect("equal")
    axes[1].set_title(f"{group_prefix} - Z variation")
    for ax in axes:
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(group_dir / f"{group_prefix}_maps.png", dpi=200)
    plt.close("all")

    return {
        "group": group_prefix, "gx": gx, "gy": gy,
        "z_mean": z_mean, "z_std": z_std,
        "valid": valid, "pts_mean": pts_mean,
        "aligned": aligned, "sheet_rms": sheet_rms,
        "closest_name": closest_name,
    }


# ──────────────────────────────────────────────
#  Compare two mean sheets (cross-group)
# ──────────────────────────────────────────────
def compare_mean_sheets(data_a, data_b, out_dir):
    na, nb = data_a["group"], data_b["group"]
    print(f"\n{'='*60}\n  COMPARISON: {na} vs {nb} (mean sheets)\n{'='*60}")

    comp_dir = out_dir / f"{na}_vs_{nb}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    pa, pb = data_a["pts_mean"], data_b["pts_mean"]

    x_lo = max(pa[:, 0].min(), pb[:, 0].min()) + GRID_SPACING
    x_hi = min(pa[:, 0].max(), pb[:, 0].max()) - GRID_SPACING
    y_lo = max(pa[:, 1].min(), pb[:, 1].min()) + GRID_SPACING
    y_hi = min(pa[:, 1].max(), pb[:, 1].max()) - GRID_SPACING
    gx, gy = np.meshgrid(np.arange(x_lo, x_hi, GRID_SPACING),
                          np.arange(y_lo, y_hi, GRID_SPACING))

    z_a = griddata(pa[:, :2], pa[:, 2], (gx, gy), method="linear")
    z_b = griddata(pb[:, :2], pb[:, 2], (gx, gy), method="linear")
    diff  = z_b - z_a
    valid = ~np.isnan(diff)
    d  = diff[valid]
    st = deviation_stats(d)

    pts_c = np.column_stack([gx[valid], gy[valid],
                              .5 * (z_a[valid] + z_b[valid])])

    print(f"  Diff ({nb}-{na}): mean={st['mean']:+.4f}, RMS={st['rms']:.4f}, "
          f"max+={st['max_pos']:+.4f}, max-={st['max_neg']:+.4f}")

    save_ply(pts_c, comp_dir / f"{na}_vs_{nb}_diff.ply",
             scalars={"diff_mm": d})

    sign = np.where(d >= 0, "positive", "negative")
    z_av, z_bv = z_a[valid], z_b[valid]
    with open(comp_dir / f"{na}_vs_{nb}_diff.csv", "w") as f:
        f.write(f"x,y,z_{na},z_{nb},diff_mm,sign\n")
        for j in range(len(d)):
            f.write(f"{pts_c[j, 0]:.4f},{pts_c[j, 1]:.4f},"
                     f"{z_av[j]:.4f},{z_bv[j]:.4f},"
                     f"{d[j]:.4f},{sign[j]}\n")

    with open(comp_dir / f"{na}_vs_{nb}_summary.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in st.items():
            f.write(f"z_{k},{v:.4f}\n")

    # Plot: diff + cross-section
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax_z = max(abs(st["max_neg"]), abs(st["max_pos"]))
    sc1 = axes[0].scatter(gx[valid], gy[valid], c=d, cmap="RdBu_r",
                           vmin=-vmax_z, vmax=vmax_z, s=.5, edgecolors="none")
    fig.colorbar(sc1, ax=axes[0], label=f"Z diff (mm): red={nb} higher")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Z: {nb} - {na}")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")

    y_mid = .5 * (y_lo + y_hi)
    band  = valid & (np.abs(gy - y_mid) < 5.0)
    if band.any():
        xp = gx[band]
        si = np.argsort(xp)
        axes[1].plot(xp[si], z_a[band][si], "b-", lw=2, label=na)
        axes[1].plot(xp[si], z_b[band][si], "r-", lw=2, label=nb)
        axes[1].fill_between(xp[si], z_a[band][si], z_b[band][si],
                              alpha=.2, color="grey")
        axes[1].legend()
        axes[1].grid(True, alpha=.3)
    axes[1].set_title("Cross-section")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Z (mm)")

    plt.suptitle(f"Mean sheets: {na} vs {nb} (RMS={st['rms']:.3f} mm)")
    plt.tight_layout()
    plt.savefig(comp_dir / f"{na}_vs_{nb}_comparison.png", dpi=200)
    plt.close("all")
    return st


# ──────────────────────────────────────────────
#  Compare closest-to-mean sheets (cross-group)
# ──────────────────────────────────────────────
def compare_closest_sheets(data_a, data_b, in_dir, out_dir):
    na, nb = data_a["closest_name"], data_b["closest_name"]
    ga, gb = data_a["group"], data_b["group"]
    print(f"\n{'='*60}\n  CLOSEST-TO-MEAN: {na} vs {nb}\n{'='*60}")

    comp_dir = out_dir / f"closest_{ga}_vs_{gb}"
    comp_dir.mkdir(parents=True, exist_ok=True)

    pts_a = load_ply(in_dir / f"{na}_noedge.ply")
    pts_b = load_ply(in_dir / f"{nb}_noedge.ply")
    lm_a, lm_b = find_landmarks(pts_a), find_landmarks(pts_b)
    R, t, rmse, _ = find_best_landmark_alignment(lm_b, lm_a)
    pts_b_al = apply_transform(pts_b, R, t)
    print(f"  Alignment RMSE: {rmse:.2f} mm")

    save_ply(pts_a,    comp_dir / f"{na}_ref.ply")
    save_ply(pts_b_al, comp_dir / f"{nb}_aligned.ply")

    x_lo = max(pts_a[:, 0].min(), pts_b_al[:, 0].min()) + GRID_SPACING
    x_hi = min(pts_a[:, 0].max(), pts_b_al[:, 0].max()) - GRID_SPACING
    y_lo = max(pts_a[:, 1].min(), pts_b_al[:, 1].min()) + GRID_SPACING
    y_hi = min(pts_a[:, 1].max(), pts_b_al[:, 1].max()) - GRID_SPACING
    gx, gy = np.meshgrid(np.arange(x_lo, x_hi, GRID_SPACING),
                          np.arange(y_lo, y_hi, GRID_SPACING))

    z_a = griddata(pts_a[:, :2],    pts_a[:, 2],    (gx, gy), method="linear")
    z_b = griddata(pts_b_al[:, :2], pts_b_al[:, 2], (gx, gy), method="linear")
    diff  = z_b - z_a
    valid = ~np.isnan(diff)
    d  = diff[valid]
    st = deviation_stats(d)

    print(f"  Diff ({nb}-{na}): mean={st['mean']:+.4f}, RMS={st['rms']:.4f}")

    pts_c = np.column_stack([gx[valid], gy[valid],
                              .5 * (z_a[valid] + z_b[valid])])
    save_ply(pts_c, comp_dir / f"closest_{ga}_vs_{gb}_diff.ply",
             scalars={"diff_mm": d})

    sign = np.where(d >= 0, "positive", "negative")
    z_av, z_bv = z_a[valid], z_b[valid]
    with open(comp_dir / f"closest_{ga}_vs_{gb}_diff.csv", "w") as f:
        f.write(f"x,y,z_{na},z_{nb},diff_mm,sign\n")
        for j in range(len(d)):
            f.write(f"{pts_c[j, 0]:.4f},{pts_c[j, 1]:.4f},"
                     f"{z_av[j]:.4f},{z_bv[j]:.4f},"
                     f"{d[j]:.4f},{sign[j]}\n")

    with open(comp_dir / f"closest_{ga}_vs_{gb}_summary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"sheet_a,{na}\nsheet_b,{nb}\n"
                f"alignment_rmse,{rmse:.4f}\n")
        for k, v in st.items():
            f.write(f"z_{k},{v:.4f}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax_z = max(abs(st["max_neg"]), abs(st["max_pos"]))
    sc1 = axes[0].scatter(gx[valid], gy[valid], c=d, cmap="RdBu_r",
                           vmin=-vmax_z, vmax=vmax_z, s=.5, edgecolors="none")
    fig.colorbar(sc1, ax=axes[0], label="Z diff (mm)")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Z: {nb} - {na}")

    y_mid = .5 * (y_lo + y_hi)
    band  = valid & (np.abs(gy - y_mid) < 5)
    if band.any():
        xp = gx[band]
        si = np.argsort(xp)
        axes[1].plot(xp[si], z_a[band][si], "b-", lw=2, label=na)
        axes[1].plot(xp[si], z_b[band][si], "r-", lw=2, label=nb)
        axes[1].fill_between(xp[si], z_a[band][si], z_b[band][si],
                              alpha=.2, color="grey")
        axes[1].legend()
        axes[1].grid(True, alpha=.3)
    axes[1].set_title("Cross-section")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Z (mm)")
    for ax in axes[:1]:
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    plt.suptitle(f"Closest-to-mean: {na} vs {nb} (RMS={st['rms']:.3f} mm)")
    plt.tight_layout()
    plt.savefig(comp_dir / f"closest_{ga}_vs_{gb}_comparison.png", dpi=200)
    plt.close("all")
    return {"alignment_rmse": rmse, "sheet_a": na, "sheet_b": nb, **st}


# ──────────────────────────────────────────────
#  Compare each sheet AND mean sheet to simulated plate
# ──────────────────────────────────────────────
def compare_to_simulated(group_data, sim_dir, out_dir):
    """
    For each group:
      1. Load the simulated PLY for the group.
      2. Align the simulated plate to the group's reference frame using
         landmarks.
      3. Compare each aligned individual sheet against the simulated plate.
      4. Compare the mean sheet against the simulated plate.
    """
    group  = group_data["group"]
    sim_file = SIM_MAP.get(group)
    if sim_file is None:
        print(f"\n  No simulated plate mapping for group {group} - skipped")
        return None

    sim_path = sim_dir / sim_file
    if not sim_path.exists():
        print(f"\n  Simulated plate not found: {sim_path} - skipped")
        return None

    print(f"\n{'='*60}")
    print(f"  SIMULATED COMPARISON: group {group} vs {sim_file}")
    print(f"{'='*60}")

    sim_dir_out = out_dir / group / "vs_simulated"
    sim_dir_out.mkdir(parents=True, exist_ok=True)

    # -- Load simulated plate and align to group's reference frame --
    pts_sim_raw = load_ply(sim_path)
    lm_sim      = find_landmarks(pts_sim_raw)

    # Reference is the first aligned sheet in the group
    ref_pts = group_data["aligned"][0]["pts"]
    ref_lm  = find_landmarks(ref_pts)

    R, t, rmse_sim, _ = find_best_landmark_alignment(lm_sim, ref_lm)
    pts_sim = apply_transform(pts_sim_raw, R, t)
    print(f"  Simulated plate alignment RMSE: {rmse_sim:.2f} mm")
    save_ply(pts_sim, sim_dir_out / f"{group}_sim_aligned.ply")

    # -- Grid the simulated plate to match group grid --
    gx, gy = group_data["gx"], group_data["gy"]
    z_sim  = griddata(pts_sim[:, :2], pts_sim[:, 2],
                      (gx, gy), method="linear")

    # -- 1. Mean sheet vs simulated --
    z_mean = group_data["z_mean"]
    diff_mean  = z_mean - z_sim
    valid_mean = ~np.isnan(diff_mean)
    d_mean  = diff_mean[valid_mean]
    st_mean = deviation_stats(d_mean)
    print(f"\n  Mean sheet vs sim: mean={st_mean['mean']:+.4f}, "
          f"RMS={st_mean['rms']:.4f}")

    pts_mean_diff = np.column_stack([gx[valid_mean], gy[valid_mean],
                                      .5 * (z_mean[valid_mean] +
                                             z_sim[valid_mean])])
    save_ply(pts_mean_diff,
             sim_dir_out / f"{group}_mean_vs_sim_diff.ply",
             scalars={"diff_mm": d_mean})

    with open(sim_dir_out / f"{group}_mean_vs_sim_diff.csv", "w") as f:
        sign = np.where(d_mean >= 0, "positive", "negative")
        f.write(f"x,y,z_mean,z_sim,diff_mm,sign\n")
        z_mv = z_mean[valid_mean]
        z_sv = z_sim[valid_mean]
        for j in range(len(d_mean)):
            f.write(f"{pts_mean_diff[j, 0]:.4f},"
                    f"{pts_mean_diff[j, 1]:.4f},"
                    f"{z_mv[j]:.4f},{z_sv[j]:.4f},"
                    f"{d_mean[j]:.4f},{sign[j]}\n")

    # -- 2. Each sheet vs simulated --
    sheet_sim_rms = {}
    summary_csv = sim_dir_out / f"{group}_sheets_vs_sim_summary.csv"
    with open(summary_csv, "w") as f:
        f.write("name,mean_dev_mm,std_dev_mm,rms_dev_mm,"
                "max_pos_mm,max_neg_mm,pct_pos,pct_neg\n")

        for i, a in enumerate(group_data["aligned"]):
            z_sheet = griddata(a["pts"][:, :2], a["pts"][:, 2],
                                 (gx, gy), method="linear")
            diff    = z_sheet - z_sim
            valid_s = ~np.isnan(diff)
            d  = diff[valid_s]
            st = deviation_stats(d)
            sheet_sim_rms[a["name"]] = st["rms"]

            save_ply(np.column_stack([gx[valid_s], gy[valid_s],
                                       z_sheet[valid_s]]),
                     sim_dir_out / f"{a['name']}_vs_sim_diff.ply",
                     scalars={"diff_mm": d})

            f.write(f"{a['name']},{st['mean']:.4f},{st['std']:.4f},"
                     f"{st['rms']:.4f},{st['max_pos']:.4f},"
                     f"{st['max_neg']:.4f},{st['pct_pos']:.1f},"
                     f"{st['pct_neg']:.1f}\n")
            print(f"    {a['name']} vs sim: RMS={st['rms']:.3f}")

    closest_to_sim = min(sheet_sim_rms, key=sheet_sim_rms.get)
    print(f"\n  * CLOSEST TO SIM: {closest_to_sim} "
          f"(RMS={sheet_sim_rms[closest_to_sim]:.4f} mm)")

    # -- 3. Plot mean sheet vs simulated --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax_z = max(abs(st_mean["max_neg"]), abs(st_mean["max_pos"]))
    if vmax_z == 0:
        vmax_z = 1.0
    sc1 = axes[0].scatter(gx[valid_mean], gy[valid_mean], c=d_mean,
                           cmap="RdBu_r", vmin=-vmax_z, vmax=vmax_z,
                           s=.5, edgecolors="none")
    fig.colorbar(sc1, ax=axes[0], label="Z diff (mm)")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"{group} mean vs sim "
                       f"(RMS={st_mean['rms']:.3f} mm)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")

    # Cross-section
    y_lo = gy[valid_mean].min()
    y_hi = gy[valid_mean].max()
    y_mid = .5 * (y_lo + y_hi)
    band = valid_mean & (np.abs(gy - y_mid) < 5)
    if band.any():
        xp = gx[band]
        si = np.argsort(xp)
        axes[1].plot(xp[si], z_mean[band][si], "b-", lw=2, label=f"{group} mean")
        axes[1].plot(xp[si], z_sim[band][si],  "r-", lw=2, label="Simulated")
        axes[1].fill_between(xp[si], z_mean[band][si], z_sim[band][si],
                              alpha=.2, color="grey")
        axes[1].legend()
        axes[1].grid(True, alpha=.3)
    axes[1].set_title("Cross-section")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Z (mm)")

    plt.suptitle(f"Group {group} mean sheet vs simulated plate")
    plt.tight_layout()
    plt.savefig(sim_dir_out / f"{group}_mean_vs_sim.png", dpi=200)
    plt.close("all")

    return {
        "group": group,
        "sim_alignment_rmse": rmse_sim,
        "mean_vs_sim":   st_mean,
        "sheet_sim_rms": sheet_sim_rms,
        "closest_to_sim": closest_to_sim,
    }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    in_dir   = Path(INPUT_DIR)
    sim_dir  = Path(SIM_DIR)
    out_dir  = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Process each group (alignment, mean sheet, deviation)
    group_data = {}
    for g in GROUPS:
        data = process_group(g, in_dir, out_dir)
        if data:
            group_data[g] = data

    # 2. Cross-group: mean sheets and closest sheets
    mean_comp, closest_comp = {}, {}
    for a, b in COMPARE_PAIRS:
        if a in group_data and b in group_data:
            mean_comp[(a, b)]    = compare_mean_sheets(
                group_data[a], group_data[b], out_dir)
            closest_comp[(a, b)] = compare_closest_sheets(
                group_data[a], group_data[b], in_dir, out_dir)
        else:
            print(f"\n  Skipping {a} vs {b}")

    # 3. Each group vs its simulated plate
    sim_comp = {}
    for g in GROUPS:
        if g in group_data:
            result = compare_to_simulated(group_data[g], sim_dir, out_dir)
            if result:
                sim_comp[g] = result

    # 4. Master CSV
    master = out_dir / "master_results.csv"
    with open(master, "w") as f:
        f.write("=== GROUPS ===\n")
        f.write("group,n_sheets,closest_to_mean,closest_rms_mm,"
                "group_mean_std_mm,group_max_std_mm\n")
        for g in sorted(group_data):
            d  = group_data[g]
            sv = d["z_std"][d["valid"]]
            f.write(f"{g},{len(d['sheet_rms'])},{d['closest_name']},"
                    f"{d['sheet_rms'][d['closest_name']]:.4f},"
                    f"{sv.mean():.4f},{sv.max():.4f}\n")

        f.write("\n=== ALL SHEETS (deviation from group mean) ===\n")
        f.write("group,sheet,mean_dev_mm,std_dev_mm,rms_dev_mm,"
                "max_pos_mm,max_neg_mm,pct_pos,pct_neg,closest_to_mean\n")
        for g in sorted(group_data):
            d = group_data[g]
            for a in d["aligned"]:
                gz   = griddata(a["pts"][:, :2], a["pts"][:, 2],
                                  (d["gx"], d["gy"]), method="linear")
                diff = (gz - d["z_mean"])[d["valid"]]
                dc   = diff[~np.isnan(diff)]
                st   = deviation_stats(dc)
                is_c = "YES" if a["name"] == d["closest_name"] else ""
                f.write(f"{g},{a['name']},{st['mean']:.4f},{st['std']:.4f},"
                        f"{st['rms']:.4f},{st['max_pos']:.4f},"
                        f"{st['max_neg']:.4f},{st['pct_pos']:.1f},"
                        f"{st['pct_neg']:.1f},{is_c}\n")

        f.write("\n=== MEAN SHEET COMPARISONS (cross-group) ===\n")
        f.write("comparison,mean_diff_mm,std_diff_mm,rms_diff_mm,"
                "max_pos_mm,max_neg_mm,pct_pos,pct_neg\n")
        for (a, b), mc in sorted(mean_comp.items()):
            f.write(f"{a}_vs_{b}_mean,{mc['mean']:.4f},{mc['std']:.4f},"
                    f"{mc['rms']:.4f},{mc['max_pos']:.4f},"
                    f"{mc['max_neg']:.4f},{mc['pct_pos']:.1f},"
                    f"{mc['pct_neg']:.1f}\n")

        f.write("\n=== CLOSEST-TO-MEAN SHEET COMPARISONS ===\n")
        f.write("comparison,sheet_a,sheet_b,alignment_rmse_mm,"
                "mean_diff_mm,std_diff_mm,rms_diff_mm,"
                "max_pos_mm,max_neg_mm,pct_pos,pct_neg\n")
        for (a, b), cc in sorted(closest_comp.items()):
            f.write(f"{a}_vs_{b}_closest,{cc['sheet_a']},{cc['sheet_b']},"
                    f"{cc['alignment_rmse']:.4f},{cc['mean']:.4f},"
                    f"{cc['std']:.4f},{cc['rms']:.4f},"
                    f"{cc['max_pos']:.4f},{cc['max_neg']:.4f},"
                    f"{cc['pct_pos']:.1f},{cc['pct_neg']:.1f}\n")

        f.write("\n=== MEAN SHEET vs SIMULATED PLATE ===\n")
        f.write("group,sim_alignment_rmse,mean_diff_mm,std_diff_mm,"
                "rms_diff_mm,max_pos_mm,max_neg_mm,"
                "pct_pos,pct_neg,closest_sheet_to_sim\n")
        for g in sorted(sim_comp):
            sc = sim_comp[g]
            st = sc["mean_vs_sim"]
            f.write(f"{g},{sc['sim_alignment_rmse']:.4f},"
                    f"{st['mean']:.4f},{st['std']:.4f},{st['rms']:.4f},"
                    f"{st['max_pos']:.4f},{st['max_neg']:.4f},"
                    f"{st['pct_pos']:.1f},{st['pct_neg']:.1f},"
                    f"{sc['closest_to_sim']}\n")

        f.write("\n=== ALL SHEETS vs SIMULATED PLATE ===\n")
        f.write("group,sheet,rms_dev_vs_sim_mm,closest_to_sim\n")
        for g in sorted(sim_comp):
            sc = sim_comp[g]
            for sheet, rms in sc["sheet_sim_rms"].items():
                is_c = "YES" if sheet == sc["closest_to_sim"] else ""
                f.write(f"{g},{sheet},{rms:.4f},{is_c}\n")

    print(f"\n{'='*60}")
    print(f"  MASTER CSV: {master}")
    print(f"  Total time: {time.time() - t0:.0f}s")
    print(f"  Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
