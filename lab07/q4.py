import os
import shutil
import subprocess

IMAGES = "images"
WORK = "colmap_workspace"
DENSE = True

db = os.path.join(WORK, "database.db")
sparse = os.path.join(WORK, "sparse")
dense = os.path.join(WORK, "dense")


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if shutil.which("colmap") is None:
    raise SystemExit("colmap install nahi hai ya PATH mein nahi")
if not os.path.isdir(IMAGES) or len(os.listdir(IMAGES)) < 3:
    raise SystemExit("images folder mein kam az kam 3 overlapping photos rakho")

os.makedirs(sparse, exist_ok=True)
os.makedirs(dense, exist_ok=True)

run(["colmap", "feature_extractor",
     "--database_path", db,
     "--image_path", IMAGES])

run(["colmap", "exhaustive_matcher",
     "--database_path", db])

run(["colmap", "mapper",
     "--database_path", db,
     "--image_path", IMAGES,
     "--output_path", sparse])

run(["colmap", "model_converter",
     "--input_path", os.path.join(sparse, "0"),
     "--output_path", os.path.join(WORK, "sparse_points.ply"),
     "--output_type", "PLY"])

if DENSE:
    run(["colmap", "image_undistorter",
         "--image_path", IMAGES,
         "--input_path", os.path.join(sparse, "0"),
         "--output_path", dense,
         "--output_type", "COLMAP"])

    run(["colmap", "patch_match_stereo",
         "--workspace_path", dense,
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true"])

    run(["colmap", "stereo_fusion",
         "--workspace_path", dense,
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", os.path.join(dense, "fused.ply")])

    run(["colmap", "poisson_mesher",
         "--input_path", os.path.join(dense, "fused.ply"),
         "--output_path", os.path.join(dense, "meshed.ply")])

try:
    import open3d as o3d
    geoms = []
    if DENSE:
        geoms.append(o3d.io.read_triangle_mesh(os.path.join(dense, "meshed.ply")))
    else:
        geoms.append(o3d.io.read_point_cloud(os.path.join(WORK, "sparse_points.ply")))
    o3d.visualization.draw_geometries(geoms)
except ImportError:
    print("open3d nahi hai. .ply files ko MeshLab ya COLMAP GUI mein kholo")
