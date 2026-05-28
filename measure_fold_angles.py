"""
measure_fold_angles.py
======================

Measures the bend angle along the curved crease of sheet-metal point clouds
(Artec Eva scans and Rhino/Stilware simulations), aggregates per subgroup,
and produces plots and CSV files.

USAGE
-----
    python measure_fold_angles.py                   # uses the paths below
    python measure_fold_angles.py --scan-dir ...    # override via CLI

CONFIGURE
---------
Edit the three DEFAULT_* paths and the parameter block (~line 40) before
running, or pass --scan-dir / --sim-dir / --output-dir on the command line.

FILE NAMING EXPECTED
--------------------
    scan_noedges/        A_1_1_noedge.ply  ...  C_1.5_5_noedge.ply
    sim_point_noedges/   A_1_noedge.ply   ...   C_5_noedge.ply

OUTPUTS  (written to output_dir/)
-------
    foldangle_collage.png          3×2 per-subgroup panel
    foldangle_master.png           all-group overview
    summary_per_subgroup.csv       mean / std / min / max + closest-to-mean files
    fold_angles_long.csv           all per-position measurements

METHOD
------
Normal-clustering + slab-based mean-normal measurement:
  1. Load PLY  →  voxel-downsample (0.5 mm)
  2. Estimate unit normals via batched k-NN PCA (k=24)
  3. Orient all normals into a common hemisphere
  4. KMeans(k=2) on normals  →  two flange populations
  5. Crease detection: vectorised mixed-label neighbourhood test
  6. Centerline: PCA  →  120 bins  →  Savitzky-Golay  →  60 samples
  7. Per sample: cut a thin slab (±SLAB_HALF_WIDTH_MM) perpendicular to
     the crease tangent; take the mean of the estimated per-point normals
     within each flange half; bend_angle = arccos(|mean_n0 · mean_n1|).
     The slab captures the full flange depth on both sides, giving the same
     angle a protractor would read.

Validation on A_1 simulation (Rhino target 20°): 19.03° measured
vs Rhino-documented 19.37–19.58°.
"""

from __future__ import annotations

# ============================================================
#  CONFIGURATION  –  edit here (or use CLI flags)
# ============================================================

DEFAULT_SCAN_DIR   = r"\\stuur02.it.liu.se\students\lovso390\Downloads\Kod\scan_noedges"
DEFAULT_SIM_DIR    = r"\\stuur02.it.liu.se\students\lovso390\Downloads\Kod\sim_point_noedges"
DEFAULT_OUTPUT_DIR = r"./foldangle_results"

TARGET_ANGLE_DEG       = 20.0   # °  — the programmed fold angle
N_SAMPLES              = 60     # measurement points along the crease
VOXEL_SIZE_MM          = 0.5    # mm — downsampling grid
# Slab half-width along the crease tangent.  Points within ±SLAB_MM of each
# sample are included from the FULL flange depth on both sides.
# Should be ~half the spacing between samples (crease_length / N_SAMPLES / 2).
# 4 mm works for plates ~300-400 mm long with 60 samples (spacing ≈ 6 mm).
SLAB_HALF_WIDTH_MM     = 4.0

# File-name patterns (regex).  Adjust if your filenames differ.
import re
SCAN_PATTERN = re.compile(
    r"^(?P<geom>[ABC])_(?P<thk>1(?:\.5)?)_(?P<rep>\d+)_noedge\.ply$",
    re.IGNORECASE,
)
SIM_PATTERN = re.compile(
    r"^(?P<geom>[ABC])_(?P<rep>\d+)(?:_.*)?\.ply$",
    re.IGNORECASE,
)

# Plot colour scheme
GEOM_COLOURS = {"A": "#1f4ea1", "B": "#2e8b57", "C": "#c0392b"}
SUBGROUP_ORDER = [("A","1"),("A","1.5"),("B","1"),("B","1.5"),("C","1"),("C","1.5")]

# ============================================================
#  IMPORTS
# ============================================================

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

# ============================================================
#  PLY LOADING
# ============================================================

def load_ply_xyz(path: str | Path) -> np.ndarray:
    """Return (N,3) float64 array of x,y,z from an ASCII or binary PLY file."""
    path = Path(path)
    with open(path, "rb") as f:
        header: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"EOF in PLY header: {path}")
            header.append(line)
            if line.strip() == b"end_header":
                break
        fmt = n_vert = None
        props: list[tuple[str,str]] = []
        in_v = False
        for raw in header:
            t = raw.decode("ascii", errors="ignore").strip()
            if t.startswith("format"):
                fmt = t.split()[1]
            elif t.startswith("element"):
                p = t.split(); in_v = p[1] == "vertex"
                if in_v: n_vert = int(p[2])
            elif t.startswith("property") and in_v:
                p = t.split()
                if p[1] == "list":
                    raise ValueError("PLY list property in vertex unsupported.")
                props.append((p[2], _ply_dtype(p[1])))
        if n_vert is None:
            raise ValueError(f"No vertex element in {path}")
        names = [n for n, _ in props]
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        if fmt == "ascii":
            arr = np.empty((n_vert, 3), dtype=np.float64)
            for i in range(n_vert):
                tok = f.readline().split()
                arr[i] = float(tok[ix]), float(tok[iy]), float(tok[iz])
            return arr
        dt = np.dtype([(n, d) for n, d in props])
        if fmt == "binary_big_endian":
            dt = dt.newbyteorder(">")
        raw_data = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt)
        return np.column_stack([raw_data["x"], raw_data["y"],
                                 raw_data["z"]]).astype(np.float64)

def _ply_dtype(t: str) -> str:
    return {"char":"i1","int8":"i1","uchar":"u1","uint8":"u1",
            "short":"i2","int16":"i2","ushort":"u2","uint16":"u2",
            "int":"i4","int32":"i4","uint":"u4","uint32":"u4",
            "float":"f4","float32":"f4","double":"f8","float64":"f8"}[t]

# ============================================================
#  VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """One point (voxel mean) per voxel cell."""
    idx = np.floor(pts / voxel_size).astype(np.int64)
    lo  = idx.min(axis=0); idx -= lo
    hi  = idx.max(axis=0) + 1
    keys = idx[:,0]*(hi[1]*hi[2]) + idx[:,1]*hi[2] + idx[:,2]
    order = np.argsort(keys)
    skeys, spts = keys[order], pts[order]
    cuts = np.where(np.diff(skeys))[0] + 1
    return np.vstack([g.mean(axis=0) for g in np.split(spts, cuts)])

# ============================================================
#  NORMAL ESTIMATION  (fully vectorised)
# ============================================================

def estimate_normals_vectorised(pts: np.ndarray, idx_knn: np.ndarray) -> np.ndarray:
    """Batched PCA normals. idx_knn: (N, k) from cKDTree.query."""
    nbh     = pts[idx_knn]                              # (N, k, 3)
    centred = nbh - pts[:, None, :]                     # (N, k, 3)
    k       = idx_knn.shape[1]
    cov     = np.einsum("nki,nkj->nij", centred, centred) / max(k-1, 1)  # (N,3,3)
    _, V    = np.linalg.eigh(cov)                       # V: (N,3,3)
    return V[:, :, 0]                                   # smallest eigenvalue

def orient_normals(normals: np.ndarray) -> np.ndarray:
    """Flip normals into a common hemisphere (in-place copy)."""
    n = normals.copy()
    _, V = np.linalg.eigh(np.cov(n.T))
    n[n @ V[:, 2] < 0] *= -1
    return n

# ============================================================
#  FLANGE SEGMENTATION
# ============================================================

def segment_flanges(normals: np.ndarray,
                     random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """KMeans(k=2) on unit normals → (labels (N,), centroids (2,3))."""
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(normals)
    c = km.cluster_centers_.copy()
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
    return labels, c

# ============================================================
#  CREASE DETECTION  (vectorised)
# ============================================================

def detect_crease(labels: np.ndarray, idx_knn: np.ndarray,
                   mixed_frac: float = 0.15) -> np.ndarray:
    """Boolean mask: True for crease-band points (vectorised)."""
    neigh_lab = labels[idx_knn]                         # (N, k)
    opp_frac  = (neigh_lab != labels[:, None]).mean(axis=1)
    return opp_frac >= mixed_frac

# ============================================================
#  CENTERLINE
# ============================================================

def build_centerline(crease_pts: np.ndarray, n_bins: int = 120,
                     sg_window: int = 9, sg_poly: int = 3,
                     n_samples: int = 60) -> np.ndarray:
    """Smooth 3-D centerline through crease band; returns (n_samples, 3)."""
    c = crease_pts - crease_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    t = crease_pts @ Vt[0]
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    verts = []
    for k in range(n_bins):
        m = (t >= edges[k]) & (t < edges[k+1])
        if m.sum() >= 3:
            verts.append(crease_pts[m].mean(axis=0))
    if len(verts) < 10:
        raise RuntimeError("Centerline: too few bins populated.")
    verts = np.array(verts)
    wl = sg_window
    if wl >= len(verts): wl = len(verts) - (1 - len(verts) % 2)
    wl = max(wl, sg_poly + 2); wl += (wl % 2 == 0)
    smoothed = np.column_stack([
        savgol_filter(verts[:, d], window_length=wl, polyorder=sg_poly)
        for d in range(3)])
    diffs = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(diffs)])
    su = np.linspace(0, s[-1], n_samples)
    return np.column_stack([np.interp(su, s, smoothed[:, d]) for d in range(3)])

# ============================================================
#  LOCAL ANGLE MEASUREMENT  (slab method — full flange depth)
# ============================================================
#
#  Why slabs instead of spherical windows:
#  For curved-crease folding the fold angle increases with distance
#  from the crease.  A spherical query centred on the crease only
#  samples the near-crease transition zone and gives angles that are
#  5-7° too small.  A thin slab cut perpendicular to the crease
#  tangent includes the FULL flange depth on both sides, giving the
#  same angle a protractor would read.

def measure_local_angles(pts: np.ndarray, normals: np.ndarray,
                          labels: np.ndarray,
                          centerline: np.ndarray,
                          slab_half_width: float,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-normal bend angle at every centerline sample.

    At each sample the slab is defined by  |dot(p - P0, T)| <= slab_half_width
    where T is the local crease tangent.  All flange points in the slab
    contribute regardless of distance from the crease.

    Returns (pos_pct, angle_deg, success_mask), each (N,).
    """
    diffs   = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    s       = np.concatenate([[0.0], np.cumsum(diffs)])
    pos_pct = 100.0 * s / s[-1]

    # Tangents via central differences
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    N         = len(centerline)
    angle_deg = np.full(N, np.nan)
    success   = np.zeros(N, dtype=bool)

    for k in range(N):
        P0 = centerline[k]
        T  = tangents[k]

        # Signed distance along the crease tangent
        along_t = (pts - P0) @ T          # (M,)
        in_slab = np.abs(along_t) <= slab_half_width

        if in_slab.sum() < 60:
            continue

        lab = labels[in_slab]
        nrm = normals[in_slab]
        if (lab == 0).sum() < 20 or (lab == 1).sum() < 20:
            continue

        n0 = nrm[lab == 0].mean(axis=0)
        n1 = nrm[lab == 1].mean(axis=0)
        l0 = np.linalg.norm(n0); l1 = np.linalg.norm(n1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        n0 /= l0; n1 /= l1

        cos_a = np.clip(abs(np.dot(n0, n1)), 0.0, 1.0)
        angle_deg[k] = np.degrees(np.arccos(cos_a))
        success[k]   = True

    TRIM = 3
    angle_deg[:TRIM] = angle_deg[-TRIM:] = np.nan
    success[:TRIM] = success[-TRIM:] = False
    return pos_pct, angle_deg, success

# ============================================================
#  MAIN MEASUREMENT FUNCTION
# ============================================================

@dataclass
class FoldAngleResult:
    file:         str
    position_pct: np.ndarray   # (N,) 0..100
    angle_deg:    np.ndarray   # (N,) NaN where failed
    success_mask: np.ndarray   # (N,) bool

def measure_file(points: np.ndarray,
                 voxel_size:       float = VOXEL_SIZE_MM,
                 k_nn:             int   = 24,
                 mixed_frac:       float = 0.15,
                 n_bins_cl:        int   = 120,
                 n_samples:        int   = N_SAMPLES,
                 slab_half_width:  float = SLAB_HALF_WIDTH_MM,
                 file_label:       str   = "",
                 ) -> FoldAngleResult:
    """Full pipeline for a single point cloud (numpy array)."""

    # 1. Downsample
    pts  = voxel_downsample(points, voxel_size)
    tree = cKDTree(pts)

    # 2. Shared k-NN lookup
    k_use = max(k_nn, 20)
    _, idx = tree.query(pts, k=k_use)

    # 3. Normals + orientation
    normals = orient_normals(estimate_normals_vectorised(pts, idx[:, :k_nn]))

    # 4. Flange segmentation
    labels, centroids = segment_flanges(normals)
    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    if min(n0, n1) / len(labels) < 0.08:
        raise RuntimeError(f"Degenerate flange split: {n0}/{n1} pts.")
    global_angle = np.degrees(np.arccos(
        np.clip(abs(np.dot(centroids[0], centroids[1])), 0, 1)))
    if global_angle < 1.0:
        raise RuntimeError(f"Global angle too small ({global_angle:.2f}°).")

    # 5. Crease detection
    is_crease = detect_crease(labels, idx[:, :20], mixed_frac)
    if is_crease.sum() < 20:
        raise RuntimeError("Too few crease points detected.")

    # 6. Centerline
    centerline = build_centerline(pts[is_crease], n_bins=n_bins_cl,
                                   n_samples=n_samples)

    # 7. Local mean-normal angles via slab method
    pos_pct, angle_deg, success = measure_local_angles(
        pts, normals, labels, centerline,
        slab_half_width=slab_half_width,
    )
    return FoldAngleResult(file=file_label, position_pct=pos_pct,
                           angle_deg=angle_deg, success_mask=success)

# ============================================================
#  FILE DISCOVERY
# ============================================================

@dataclass
class ScanFile:
    path: Path; geom: str; thickness: str; replicate: int
    @property
    def subgroup(self): return f"{self.geom}_{self.thickness}"
    @property
    def label(self):    return f"{self.subgroup}_{self.replicate}"

@dataclass
class SimFile:
    path: Path; geom: str; replicate: int
    @property
    def label(self): return f"{self.geom}_{self.replicate}"

def discover_scans(scan_dir: Path) -> list[ScanFile]:
    out = []
    if not scan_dir.exists(): return out
    for p in sorted(scan_dir.iterdir()):
        m = SCAN_PATTERN.match(p.name)
        if m:
            out.append(ScanFile(p, m.group("geom").upper(),
                                m.group("thk"), int(m.group("rep"))))
    return out

def discover_sims(sim_dir: Path) -> list[SimFile]:
    out = []
    if not sim_dir.exists(): return out
    for p in sorted(sim_dir.iterdir()):
        if p.suffix.lower() != ".ply": continue
        m = SIM_PATTERN.match(p.name)
        if m:
            out.append(SimFile(p, m.group("geom").upper(), int(m.group("rep"))))
    return out

# ============================================================
#  MEASUREMENT CACHE
# ============================================================

def measure_one(path: Path,
                 n_samples: int, voxel_size: float,
                 slab_half_width: float) -> FoldAngleResult:
    """Measure a single PLY file — no caching."""
    print(f"  measuring {path.name} ...", flush=True)
    t0  = time.time()
    pts = load_ply_xyz(path)
    res = measure_file(pts, voxel_size=voxel_size,
                       slab_half_width=slab_half_width,
                       n_samples=n_samples,
                       file_label=path.name)
    print(f"    {time.time()-t0:.1f}s  "
          f"({res.success_mask.sum()}/{n_samples} samples ok)")
    return res

# ============================================================
#  AGGREGATION
# ============================================================

def stack_curves(results: list[FoldAngleResult]) -> np.ndarray:
    return np.stack([r.angle_deg for r in results], axis=0)

def closest_to_mean(curves: np.ndarray,
                     labels: list[str]) -> tuple[str, np.ndarray]:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_c = np.nanmean(curves, axis=0)
        rmse   = np.sqrt(np.nanmean((curves - mean_c[None,:]) ** 2, axis=1))
    return labels[int(np.argmin(rmse))], rmse

# ============================================================
#  PLOTTING
# ============================================================

def _blue_shades(n: int) -> list:
    return [plt.get_cmap("Blues")(0.35 + 0.55*i/max(1,n-1)) for i in range(n)]

def plot_subgroup(ax_c, ax_h, pos, scan_curves, scan_labels,
                  sim_curve, sim_label, target):
    shades = _blue_shades(scan_curves.shape[0])
    for i in range(scan_curves.shape[0]):
        ax_c.plot(pos, scan_curves[i], "-o", ms=2.5, lw=1.0,
                  color=shades[i], label=f"scan: {scan_labels[i]}")
    if sim_curve is not None:
        ax_c.plot(pos, sim_curve, "-s", ms=3, lw=1.2,
                  color="#3a2a5f", label=f"sim: {sim_label}")
    ax_c.axhline(target, color="crimson", ls="--", lw=1.0,
                 label=f"Target ({target:.0f}" + chr(176) + ")")
    ax_c.set_xlabel("Position along crease (%)")
    ax_c.set_ylabel("Bend angle " + chr(952) + " (" + chr(176) + ")")
    ax_c.set_xlim(-2, 102); ax_c.set_ylim(7, 22)
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(fontsize=7, loc="lower right")
    bins = np.linspace(7, 22, 31)
    ax_h.hist(scan_curves[~np.isnan(scan_curves)], bins=bins,
              orientation="horizontal", color="#5b8dd9", alpha=0.75)
    if sim_curve is not None:
        ax_h.hist(sim_curve[~np.isnan(sim_curve)], bins=bins,
                  orientation="horizontal", color="#6a4aaa", alpha=0.85)
    ax_h.set_xlabel("Count"); ax_h.set_ylim(7, 22)
    ax_h.grid(True, alpha=0.25); ax_h.set_yticklabels([])

def make_collage(grouped, sims_info, out_path, target):
    fig = plt.figure(figsize=(18, 16))
    outer = GridSpec(3, 2, figure=fig, wspace=0.18, hspace=0.42)
    for slot, (geom, thk) in enumerate(SUBGROUP_ORDER):
        sg   = f"{geom}_{thk}"
        info = grouped.get(sg)
        if info is None or info["curves"].size == 0: continue
        vals = info["curves"][~np.isnan(info["curves"])]
        mean = vals.mean() if len(vals) else float("nan")
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        mn   = vals.min() if len(vals) else float("nan")
        mx   = vals.max() if len(vals) else float("nan")
        inner = outer[slot].subgridspec(2, 1, height_ratios=[0.15, 1.0], hspace=0.05)
        ax_hd = fig.add_subplot(inner[0]); ax_hd.axis("off")
        ax_hd.text(0.5, 1.05, sg, ha="center", va="bottom",
                   fontsize=15, weight="bold", transform=ax_hd.transAxes)
        for j, (lbl, val) in enumerate([
            ("Target", f"{target:.2f}" + chr(176)),
            ("Scan mean", f"{mean:.2f}" + chr(176)),
            ("Scan std. deviation", f"{std:.3f}" + chr(176)),
            ("Scan min \u2013 max", f"{(mx-mn):.2f}" + chr(176)),
        ]):
            x = (j + 0.5) / 4
            ax_hd.text(x, 0.65, lbl, ha="center", va="bottom",
                       fontsize=10, color="#666", transform=ax_hd.transAxes)
            ax_hd.text(x, 0.05, val, ha="center", va="bottom",
                       fontsize=14, color="#222", transform=ax_hd.transAxes)
        pg  = inner[1].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.05)
        ax_c = fig.add_subplot(pg[0]); ax_h = fig.add_subplot(pg[1], sharey=ax_c)
        si   = sims_info.get(geom)
        plot_subgroup(ax_c, ax_h, info["pos"], info["curves"], info["labels"],
                      si["curve"] if si else None,
                      si["label"] if si else None, target)
    fig.suptitle("Per-subgroup fold-angle curves (scans + simulation)",
                 fontsize=16, y=0.995)
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)

def make_master(grouped, sims_info, out_path, target):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(target, color="#8a3a3a", ls="--", lw=1.2,
               label=f"Target ({target:.0f}" + chr(176) + ")")
    for geom, thk in SUBGROUP_ORDER:
        sg   = f"{geom}_{thk}"
        info = grouped.get(sg)
        if info is None or info["curves"].size == 0: continue
        c    = GEOM_COLOURS[geom]
        ls   = "-" if thk == "1" else "--"
        mean = np.nanmean(info["curves"], axis=0)
        ax.fill_between(info["pos"],
                        np.nanmin(info["curves"], axis=0),
                        np.nanmax(info["curves"], axis=0),
                        color=c, alpha=0.18, lw=0)
        ax.plot(info["pos"], mean, color=c, ls=ls, lw=1.6,
                label=f"scan {sg} (n={info['curves'].shape[0]})")
    for geom, si in sims_info.items():
        c = GEOM_COLOURS[geom]
        ax.plot(si["pos"], si["curve"], color=c, ls="-", lw=1.0,
                marker="s", mfc="white", mec=c, ms=4.5,
                label=f"sim {geom} ({si['label']})")
    ax.set_xlabel("Position along crease (%)")
    ax.set_ylabel("Bend angle " + chr(952) + " (" + chr(176) + ")")
    ax.set_title("All curves: scans (mean " + chr(177) + " range band) vs simulations")
    ax.set_ylim(7, 22); ax.set_xlim(-2, 102); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-dir",    default=DEFAULT_SCAN_DIR)
    parser.add_argument("--sim-dir",     default=DEFAULT_SIM_DIR)
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-angle",type=float, default=TARGET_ANGLE_DEG)
    parser.add_argument("--n-samples",   type=int,   default=N_SAMPLES)
    parser.add_argument("--voxel-size",  type=float, default=VOXEL_SIZE_MM)
    args = parser.parse_args()

    scan_dir  = Path(args.scan_dir)
    sim_dir   = Path(args.sim_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scan dir : {scan_dir}")
    print(f"Sim dir  : {sim_dir}")
    print(f"Output   : {out_dir}")

    scans = discover_scans(scan_dir)
    sims  = discover_sims(sim_dir)
    print(f"\nFound {len(scans)} scan files, {len(sims)} sim files")
    if not scans and not sims:
        print("Nothing to do."); return

    ckw = dict(n_samples=args.n_samples, voxel_size=args.voxel_size,
               slab_half_width=SLAB_HALF_WIDTH_MM)

    # --- measure ---
    print("\n--- Measuring scans ---")
    scan_res: dict[str, FoldAngleResult] = {}
    for sf in scans:
        try:
            scan_res[sf.label] = measure_one(sf.path, **ckw)
        except Exception as e:
            print(f"  SKIPPED {sf.path.name}: {e}")

    print("\n--- Measuring simulations ---")
    sim_res: dict[str, FoldAngleResult] = {}   # key = sm.label  e.g. "A_1"
    for sm in sims:
        try:
            sim_res[sm.label] = measure_one(sm.path, **ckw)
        except Exception as e:
            print(f"  SKIPPED {sm.path.name}: {e}")

    # --- aggregate per scan subgroup ---
    grouped: dict[str, dict] = {}
    for geom, thk in SUBGROUP_ORDER:
        sg      = f"{geom}_{thk}"
        members = sorted([s for s in scans if s.subgroup == sg],
                         key=lambda s: s.replicate)
        valid   = [m for m in members if m.label in scan_res]
        if not valid:
            grouped[sg] = {"pos": np.array([]), "curves": np.empty((0,0)),
                           "labels": []}
            continue
        results = [scan_res[m.label] for m in valid]
        grouped[sg] = {"pos":    results[0].position_pct,
                       "curves": stack_curves(results),
                       "labels": [m.label for m in valid]}

    # --- aggregate sim replikaten per geometri → mean curve ---
    sims_info: dict[str, dict] = {}
    for geom in sorted({sm.geom for sm in sims}):
        geom_sims = sorted([sm for sm in sims if sm.geom == geom],
                            key=lambda s: s.replicate)
        valid_sims = [sm for sm in geom_sims if sm.label in sim_res]
        if not valid_sims:
            continue
        sim_results = [sim_res[sm.label] for sm in valid_sims]
        sim_curves  = stack_curves(sim_results)          # (n_rep, 60)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sim_mean = np.nanmean(sim_curves, axis=0)     # (60,)
        sims_info[geom] = {
            "pos":        sim_results[0].position_pct,
            "curve":      sim_mean,                      # mean of all replikaten
            "curves_all": sim_curves,                    # all individual curves
            "labels":     [sm.label for sm in valid_sims],
            "n":          len(valid_sims),
            "label":      f"sim {geom} mean (n={len(valid_sims)})",
        }

    # --- closest-to-mean identification + summary CSV ---
    rows = []
    for geom, thk in SUBGROUP_ORDER:
        sg   = f"{geom}_{thk}"
        info = grouped[sg]
        if info["curves"].size == 0: continue

        # Which scan is closest to the scan group mean?
        best_scan_grp, _ = closest_to_mean(info["curves"], info["labels"])

        # Which scan is closest to the sim mean?
        best_scan_sim = ""
        if geom in sims_info:
            sc = sims_info[geom]["curve"]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                rmse = np.sqrt(np.nanmean(
                    (info["curves"] - sc[None,:]) ** 2, axis=1))
            best_scan_sim = info["labels"][int(np.argmin(rmse))]

        # Which sim replikat is closest to the sim group mean?
        best_sim_grp = ""
        if geom in sims_info and sims_info[geom]["n"] > 1:
            sim_labels  = sims_info[geom]["labels"]
            sim_curves  = sims_info[geom]["curves_all"]
            best_sim_grp, _ = closest_to_mean(sim_curves, sim_labels)

        vals = info["curves"][~np.isnan(info["curves"])]
        rows.append(dict(
            subgroup             = sg,
            geometry             = geom,
            thickness_mm         = thk,
            n_replicates         = info["curves"].shape[0],
            target_deg           = args.target_angle,
            scan_mean_deg        = float(vals.mean())      if len(vals) else float("nan"),
            scan_std_deg         = float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
            scan_min_deg         = float(vals.min())        if len(vals) else float("nan"),
            scan_max_deg         = float(vals.max())        if len(vals) else float("nan"),
            sim_mean_deg         = float(np.nanmean(sims_info[geom]["curve"]))
                                   if geom in sims_info else float("nan"),
            sim_n                = sims_info[geom]["n"] if geom in sims_info else 0,
            scan_closest_to_scan_mean = best_scan_grp,
            scan_closest_to_sim_mean  = best_scan_sim,
            sim_closest_to_sim_mean   = best_sim_grp,
        ))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary_per_subgroup.csv", index=False)
    print(f"\nWrote summary_per_subgroup.csv")

    # --- long-format CSV (scans + all individual sim replikat) ---
    long_rows = []
    for sf in scans:
        if sf.label not in scan_res: continue
        r = scan_res[sf.label]
        for p, a, ok in zip(r.position_pct, r.angle_deg, r.success_mask):
            long_rows.append(dict(kind="scan", geometry=sf.geom,
                                  thickness_mm=sf.thickness,
                                  replicate=sf.replicate, file=r.file,
                                  position_pct=float(p), angle_deg=float(a),
                                  success=bool(ok)))
    for sm in sims:
        if sm.label not in sim_res: continue
        r = sim_res[sm.label]
        for p, a, ok in zip(r.position_pct, r.angle_deg, r.success_mask):
            long_rows.append(dict(kind="sim", geometry=sm.geom,
                                  thickness_mm="", replicate=sm.replicate,
                                  file=r.file,
                                  position_pct=float(p), angle_deg=float(a),
                                  success=bool(ok)))
    pd.DataFrame(long_rows).to_csv(out_dir / "fold_angles_long.csv", index=False)
    print("Wrote fold_angles_long.csv")

    # --- plots ---
    make_collage(grouped, sims_info, out_dir / "foldangle_collage.png",
                 args.target_angle)
    print("Wrote foldangle_collage.png")
    make_master(grouped, sims_info, out_dir / "foldangle_master.png",
                args.target_angle)
    print("Wrote foldangle_master.png")

    # --- console summary ---
    print("\n=== Summary ===")
    if not summary_df.empty:
        with pd.option_context("display.width", 180, "display.max_columns", None):
            print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
