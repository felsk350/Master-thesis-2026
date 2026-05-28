"""
trim_sheets.py
=================
Förbehandlar scannade och simulerade plåtar.

Skanningar (mesh eller punktmoln):
  1. Sampla mesh -> punktmoln (eller behåll punktmoln som det är)
  2. Ta bort punkter inom BOUNDARY_MARGIN_SCAN mm från konvexa höljet
     (vertex-baserad filtrering — räcker för scans eftersom enda målet
      är att radera flygpixlar; ingen mesh-output)
  3. Spara som punktmoln

Simuleringar (mesh in, behåll mesh-information):
  1. Trimma mesh CLEAN med plane-slice mot offset av 2D konvex hull
     (margin = BOUNDARY_MARGIN_SIM mm). Detta skapar nya vertices exakt
     på snittet, så kanten blir inte jagged.
  2. Spara den trimmade mesh:en  -> SIM_MESH_OUT_DIR
  3. Sampla yt-punkter från den trimmade meshen (area-viktad,
     reproducerbar via seed) -> SIM_POINT_OUT_DIR

Dependencies:
    pip install plyfile numpy scipy trimesh shapely

Usage:
    python trim_sheets.py
"""

import numpy as np
from scipy.spatial import ConvexHull
from plyfile import PlyData, PlyElement
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

import trimesh
from shapely.geometry import Polygon

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
# --- Skanningar ---
SCAN_IN_DIR        = r"Z:\lovso390\Downloads\Kod\scans"
SCAN_OUT_DIR       = r"Z:\lovso390\Downloads\Kod\scans_no_edges"
BOUNDARY_MARGIN_SCAN = 10.0   # mm (för att radera flygpixlar)
SCAN_SUFFIX        = "noedge"

# --- Simuleringar ---
SIM_IN_DIR         = r"\\stuur02.it.liu.se\students\lovso390\Downloads\Kod\Sim_all"
SIM_MESH_OUT_DIR   = r"Z:\lovso390\Downloads\Kod\sim_mesh_noedges"
SIM_POINT_OUT_DIR  = r"Z:\lovso390\Downloads\Kod\sim_point_noedges"
BOUNDARY_MARGIN_SIM = 10.0     # mm (clean cut, ingen taggig kant)
SIM_MESH_SUFFIX    = "mesh_noedge"
SIM_POINT_SUFFIX   = "noedge"

# --- Punktsampling ---
TARGET_N_POINTS    = 917_813  # medeltäthet från scannade ark
SAMPLE_SEED        = 42       # reproducerbarhet
N_WORKERS          = min(cpu_count(), 8)


# ──────────────────────────────────────────────
#  I/O-hjälpare
# ──────────────────────────────────────────────
def save_ply_points(pts, path):
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    arr = np.empty(len(pts), dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=True).write(str(path))


def extract_faces_plyfile(face_data):
    """Robust extrahering av face-index från plyfile."""
    if hasattr(face_data[0], "__len__"):
        try:
            return np.vstack(face_data["vertex_indices"])
        except (ValueError, IndexError):
            return np.array([f[0] for f in face_data])
    return np.array([f[0] for f in face_data])


def load_ply_points_and_faces(path):
    """Returnerar (vertices Nx3, faces Mx3 eller None)."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    verts = np.column_stack([v["x"], v["y"], v["z"]])
    faces = None
    if "face" in ply:
        faces = extract_faces_plyfile(ply["face"].data)
    return verts, faces


# ──────────────────────────────────────────────
#  Clean mesh trim (plane-slice mot offset hull)
# ──────────────────────────────────────────────
def trim_mesh_clean(mesh, margin_mm):
    """
    Trimma trimesh.Trimesh genom att slice:a mot en inåt-offsettad
    polygon av meshens 2D konvexa hölje. Skapar nya vertices exakt på
    snittet, så kanten blir helt jämn (inte vertex-jagged).

    Förutsätter att meshen projicerar förnuftigt till XY (gäller för
    plåtar med < ca 85° lokal lutning mot Z-axeln).
    """
    verts2d = mesh.vertices[:, :2]
    hull = ConvexHull(verts2d)
    boundary = Polygon(verts2d[hull.vertices])
    inset = boundary.buffer(-margin_mm)
    if inset.is_empty or not inset.is_valid:
        raise ValueError(
            f"Offset-polygon tom/ogiltig för margin {margin_mm} mm"
        )

    # CCW så att inward-normal pekar inåt
    coords = np.array(inset.exterior.coords[:-1])
    if not inset.exterior.is_ccw:
        coords = coords[::-1]

    result = mesh.copy()
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        edge = p2 - p1
        n2d = np.array([-edge[1], edge[0]])
        n2d /= np.linalg.norm(n2d)
        plane_normal = np.array([n2d[0], n2d[1], 0.0])
        plane_origin = np.array([p1[0], p1[1], 0.0])
        result = trimesh.intersections.slice_mesh_plane(
            result, plane_normal, plane_origin
        )
    # Städa bort eventuella degenererade trianglar från snitt
    result.merge_vertices(digits_vertex=2)              # slå ihop inom 0,01 mm
    result.update_faces(result.nondegenerate_faces(height=1e-3))  # ta bort < 0,001 mm²
    result.remove_unreferenced_vertices()
    return result


# ──────────────────────────────────────────────
#  Vertex-baserad kant-filtrering (för scans)
# ──────────────────────────────────────────────
def compute_boundary_mask_points(pts, margin):
    """Vektoriserad kantfiltrering med chunking för stora punktmoln."""
    centroid = pts.mean(axis=0)
    cov = np.cov(pts - centroid, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    u, v = eigvecs[:, 2], eigvecs[:, 1]
    pts_2d = np.column_stack([(pts - centroid) @ u, (pts - centroid) @ v])

    hull = ConvexHull(pts_2d)
    hp = pts_2d[hull.vertices]
    a, b = hp, np.roll(hp, -1, axis=0)
    ab = b - a
    ab2 = np.sum(ab ** 2, axis=1)

    chunk = 100_000
    dmin = np.empty(len(pts_2d))
    for s in range(0, len(pts_2d), chunk):
        e = min(s + chunk, len(pts_2d))
        ch = pts_2d[s:e]
        ap = ch[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) /
                    (ab2[None, :] + 1e-12), 0, 1)
        cl = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dmin[s:e] = np.linalg.norm(ch[:, None, :] - cl, axis=2).min(axis=1)

    return dmin > margin


# ──────────────────────────────────────────────
#  Mesh-yt-sampling (reproducerbar)
# ──────────────────────────────────────────────
def sample_mesh_surface(mesh, n_samples, seed=SAMPLE_SEED):
    """Area-viktad barycentrisk sampling via trimesh, seedad RNG."""
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    return np.asarray(pts)


# ──────────────────────────────────────────────
#  Processa en SCAN-fil
# ──────────────────────────────────────────────
def process_scan(args):
    """args: (ply_path, out_dir, suffix, margin, target_n)"""
    ply_path, out_dir, suffix, margin, target_n = args
    t0 = time.time()

    verts, faces = load_ply_points_and_faces(ply_path)
    n_orig = len(verts)

    # Om scan är en mesh: sampla ytan till target_n; annars behåll punkter
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts = sample_mesh_surface(mesh, target_n)
        info = f"mesh->sampled {target_n}"
    else:
        pts = verts
        info = f"pc (full {n_orig})"

    # Vertex-baserad kantfilter (10 mm bort flygpixlar)
    mask = compute_boundary_mask_points(pts, margin)
    trimmed = pts[mask]
    removed = len(pts) - len(trimmed)

    out_path = out_dir / f"{ply_path.stem}_{suffix}.ply"
    save_ply_points(trimmed, out_path)

    return ("scan", ply_path.name, n_orig, len(pts), len(trimmed),
            removed, info, time.time() - t0)


# ──────────────────────────────────────────────
#  Processa en SIM-fil (mesh in, mesh + punkter ut)
# ──────────────────────────────────────────────
def process_sim(args):
    """args: (ply_path, mesh_out_dir, point_out_dir,
              mesh_suffix, point_suffix, margin, target_n)

    De två grenarna är nu oberoende, precis som i flödesschemat:

      Gren A – mesh:
        original mesh -> trim_mesh_clean (plane-slice) -> spara trimmad mesh

      Gren B – punktmoln:
        original mesh -> sampla target_n punkter (full yta)
                      -> compute_boundary_mask_points (vertex-filter, som scans)
                      -> spara filtrerat punktmoln

    Eftersom sampling sker FÖRE kantborttagning innehåller punktmolnet
    fler punkter än tidigare (ingen area förloras till plane-slice-marginalerna
    innan samplingen).
    """
    (ply_path, mesh_out_dir, point_out_dir,
     mesh_suffix, point_suffix, margin, target_n) = args
    t0 = time.time()

    # Ladda som trimesh
    mesh = trimesh.load(str(ply_path), process=False, force="mesh")
    if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
        return ("sim", ply_path.name, 0, 0, 0, 0,
                "SKIP (no faces)", time.time() - t0)
    n_verts_orig = len(mesh.vertices)
    n_faces_orig = len(mesh.faces)

    # ── Gren A: Trim mesh clean (plane-slice) ────────────────────────────
    trimmed = trim_mesh_clean(mesh, margin)
    n_verts_trim = len(trimmed.vertices)
    n_faces_trim = len(trimmed.faces)

    mesh_out = mesh_out_dir / f"{ply_path.stem}_{mesh_suffix}.ply"
    trimmed.export(str(mesh_out))

    # ── Gren B: Sampla från ORIGINAL-meshen, sedan vertex-baserat kantfilter
    #    (samma logik som för scans — margin avser avstånd till konvext hölje
    #     i 2D-projektionen, inte plane-slice)  ────────────────────────────
    pts_full = sample_mesh_surface(mesh, target_n)
    mask = compute_boundary_mask_points(pts_full, margin)
    pts_trimmed = pts_full[mask]
    n_removed = len(pts_full) - len(pts_trimmed)

    pc_out = point_out_dir / f"{ply_path.stem}_{point_suffix}.ply"
    save_ply_points(pts_trimmed, pc_out)

    info = (f"mesh {n_verts_orig}->{n_verts_trim} verts, "
            f"{n_faces_orig}->{n_faces_trim} faces | "
            f"pc: sampled {len(pts_full)} -> kant -{n_removed} -> {len(pts_trimmed)} pts")
    return ("sim", ply_path.name, n_verts_orig, n_verts_trim,
            len(pts_trimmed), n_removed, info, time.time() - t0)


# ──────────────────────────────────────────────
#  Job-runners
# ──────────────────────────────────────────────
def run_scan_job():
    in_dir  = Path(SCAN_IN_DIR)
    out_dir = Path(SCAN_OUT_DIR)
    print("\n" + "=" * 70)
    print(f"  SCAN-JOBB: {in_dir}")
    print(f"  ->         {out_dir}")
    print(f"  margin={BOUNDARY_MARGIN_SCAN} mm, target={TARGET_N_POINTS} pts")
    print("=" * 70)

    if not in_dir.exists():
        print(f"  [HOPPAR ÖVER] Input-mappen finns inte: {in_dir}")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.ply"))
    print(f"  Hittade {len(files)} PLY-filer, workers={N_WORKERS}\n")
    if not files:
        return []

    args = [(f, out_dir, SCAN_SUFFIX, BOUNDARY_MARGIN_SCAN, TARGET_N_POINTS)
            for f in files]
    t = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_scan, args)
    for _, name, n_orig, n_in, n_out, removed, info, dt in results:
        pct = 100 * removed / n_in if n_in else 0
        print(f"  {name}: {info}, kant -{removed} ({pct:.1f}%), "
              f"kvar {n_out}  [{dt:.1f}s]")
    print(f"\n  Klart: {len(results)} filer på {time.time() - t:.1f}s")
    return results


def run_sim_job():
    in_dir       = Path(SIM_IN_DIR)
    mesh_out_dir = Path(SIM_MESH_OUT_DIR)
    point_out_dir = Path(SIM_POINT_OUT_DIR)
    print("\n" + "=" * 70)
    print(f"  SIM-JOBB:   {in_dir}")
    print(f"  ->  mesh:   {mesh_out_dir}")
    print(f"  ->  points: {point_out_dir}")
    print(f"  margin={BOUNDARY_MARGIN_SIM} mm (clean cut), "
          f"target={TARGET_N_POINTS} pts")
    print("=" * 70)

    if not in_dir.exists():
        print(f"  [HOPPAR ÖVER] Input-mappen finns inte: {in_dir}")
        return []
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    point_out_dir.mkdir(parents=True, exist_ok=True)
    # Rekursiv sökning så att .ply-filer i undermappar (A, B, C, ...) tas med
    files = sorted(in_dir.rglob("*.ply"))
    print(f"  Hittade {len(files)} PLY-filer (rekursivt), workers={N_WORKERS}\n")
    if not files:
        return []

    # Bevara gruppstrukturen (A/B/C) i output-mapparna så filnamn inte krockar
    args = []
    for f in files:
        rel_parent = f.parent.relative_to(in_dir)     # "A", "B", "C", eller "."
        m_dir = mesh_out_dir / rel_parent
        p_dir = point_out_dir / rel_parent
        m_dir.mkdir(parents=True, exist_ok=True)
        p_dir.mkdir(parents=True, exist_ok=True)
        args.append((f, m_dir, p_dir,
                     SIM_MESH_SUFFIX, SIM_POINT_SUFFIX,
                     BOUNDARY_MARGIN_SIM, TARGET_N_POINTS))

    t = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_sim, args)
    for arg, result in zip(args, results):
        rel = arg[0].relative_to(in_dir)
        info, dt = result[6], result[7]
        print(f"  {rel}: {info}  [{dt:.1f}s]")
    print(f"\n  Klart: {len(results)} filer på {time.time() - t:.1f}s")
    return results


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    t_total = time.time()
    print(f"BOUNDARY_MARGIN_SCAN = {BOUNDARY_MARGIN_SCAN} mm")
    print(f"BOUNDARY_MARGIN_SIM  = {BOUNDARY_MARGIN_SIM} mm")
    print(f"TARGET_N_POINTS      = {TARGET_N_POINTS}")
    print(f"SAMPLE_SEED          = {SAMPLE_SEED}")
    print(f"N_WORKERS            = {N_WORKERS}")

    scan_results = run_scan_job()
    sim_results  = run_sim_job()

    print("\n" + "=" * 70)
    print(f"  ALLT KLART: {len(scan_results)} scans + "
          f"{len(sim_results)} sims  på {time.time() - t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
