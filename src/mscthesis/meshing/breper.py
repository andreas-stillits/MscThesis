""" 
npy_to_brep.py
Convert a voxel representation (3D numpy array .npy) to a BREP file using FreeCAD.

The FreeCAD conversion is done by writing a temporary Python script 
that FreeCAD executes in headless mode. The input filename is passed 
via the environment variable 'input_path'.

The code will produce an .stl and a .brep file in the same directory
as the input .npy file and inherit its file name.

Usage:
    python npy_to_brep.py <input_file.npy> [options]

Options:
    --spacing sx,sy,sz       Voxel spacing in each dimension (default 1.0,1.0,1.0)
    --freecad-cmd PATH       Path to the FreeCAD command line tool (default: freecadcmd-daily)
    --freecad-script PATH    Path to a custom FreeCAD conversion script (default assumes script is in the same directory as this file)
    --suppress               Suppress verbose output
    --smoothing_iter N       Number of Taubin smoothing iterations (default 10)
    --decimate N             Target number of triangles after decimation (default 10,000)
    --open-gui               Open Open3D visualization window (default: False)
"""

import os
import subprocess
import argparse
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import open3d
from skimage import measure
from mscthesis.utilities.reporter import Reporter
from mscthesis.utilities.checker import check_io_paths


DEFAULT_FREECAD_CMD = "freecadcmd-daily"
DEFAULT_FREECAD_SCRIPT = os.path.join(os.path.dirname(__file__), "freecad_converter.py")
DEFAULT_SMOOTHING_ITER = 15
DEFAULT_DECIMATE = 10_000
SHRINKAGE_TOLERANCE = 0.10  # 10 percent volume/surface shrinkage tolerance


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert water tight voxel representation in .npy to CAD model in .brep")
    p.add_argument("input_path", type=str, help="Path to the input .npy file containing a 3D array (boolean or 0/1)")
    p.add_argument("--output-path", type=str, default=None, help="Path to output .brep file. If not provided, will use input path with .brep extension")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress verbose output")
    p.add_argument("--spacing", type=lambda s: tuple(float(x) for x in s.split(',')), default=(1.0,1.0,1.0), help="voxel spacing sx,sy,sz (default 1.0,1.0,1.0)")
    p.add_argument("--freecad-cmd", default=DEFAULT_FREECAD_CMD, help="Path to the FreeCAD command line tool")
    p.add_argument("--freecad-script", default=DEFAULT_FREECAD_SCRIPT, help=f"Path to a custom FreeCAD conversion script (default assumes script is in the same directory as {__file__})") 
    p.add_argument("--smoothing-iter", type=int, default=DEFAULT_SMOOTHING_ITER, help=f"Number of Taubin smoothing iterations (default {DEFAULT_SMOOTHING_ITER})")
    p.add_argument("--decimate", type=int, default=DEFAULT_DECIMATE, help=f"Target number of triangles after decimation (default {DEFAULT_DECIMATE})")
    p.add_argument("--shrinkage-tolerance", type=float, default=SHRINKAGE_TOLERANCE, help=f"Maximum allowed surface/volume shrinkage (default {100 * SHRINKAGE_TOLERANCE:.1f} %%)") # 2 percent signs do to argparse formatting standards. Will read as one
    p.add_argument("--open-gui", default=False, action="store_true", help="Open Open3D visualization window (default: False)")
    args = p.parse_args(argv)
    
    # check if input file exists, has right extension and check/derive output path
    check_io_paths(args, ".npy", ".brep")

    # initialize reporter
    reporter = Reporter(args, __file__)
    reporter.start_log()

    # load file
    voxels = np.load(args.input_path)
    reporter.print(f"loaded volume shape: {voxels.shape}, spacing: {args.spacing}")

    # convert to triangular surface mesh using Open3D
    verts, faces, normals, values = measure.marching_cubes(voxels, spacing=args.spacing, level=0.5)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(verts)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh = clean_mesh(mesh)
    # get surface area and volume
    pre_area = mesh.get_surface_area()
    pre_volume = mesh.get_volume()
    reporter.print(f"Initial triangles = {len(np.asarray(mesh.triangles))}, ")
    reporter.print(f"vertices = {len(np.asarray(mesh.vertices))}, ")
    reporter.print(f"surface area = {pre_area:.2f}, ")
    reporter.print(f"volume = {pre_volume:.2f}")

    # apply smoothing
    reporter.print(f"Applying {args.smoothing_iter} Taubin smoothing iterations...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smoothing_iter)
    mesh = clean_mesh(mesh)

    # apply mesh decimation
    reporter.print(f"Decimating mesh to ~{args.decimate} triangles...")
    current = len(np.asarray(mesh.triangles)) # get current triangle count
    target = args.decimate
    if not target >= current:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh = clean_mesh(mesh)
    else: 
        reporter.print(f"Skipping decimation as current triangle count {current} is less than target {target}.")
    post_area = mesh.get_surface_area()
    post_volume = mesh.get_volume()
    reporter.print(f"Post-decimation triangles = {len(np.asarray(mesh.triangles))}, ")
    reporter.print(f"vertices = {len(np.asarray(mesh.vertices))}, ")
    reporter.print(f"surface area = {post_area:.2f}, ")
    reporter.print(f"volume = {post_volume:.2f}")

    # check that mesh is manifold and water tight
    edge_manifold = mesh.is_edge_manifold()
    vertex_manifold = mesh.is_vertex_manifold()
    watertight = mesh.is_watertight()
    reporter.print(f"Edge manifold: {edge_manifold}")
    reporter.print(f"Vertex manifold: {vertex_manifold}")
    reporter.print(f"Watertight: {watertight}")

    # abort if these conditions are not met
    if not (edge_manifold and vertex_manifold and watertight):
        reporter.print("Error: Mesh is not manifold and watertight. Cannot convert to BREP.")
        # save as .stl for inspection
        stl_path = os.path.splitext(args.output_path)[0] + ".stl"
        written = open3d.io.write_triangle_mesh(stl_path, mesh)
        if not written:
            raise RuntimeError(f"Failed to write STL file to {stl_path}")
        reporter.print(f"Saved non-manifold mesh as {stl_path} for inspection.")
        
    else:
        # given that the mesh is manifold and water tight, save as .stl and proceed to BREP conversion
        stl_path = os.path.splitext(args.output_path)[0] + ".stl"
        written = open3d.io.write_triangle_mesh(stl_path, mesh)
        if not written:
            raise RuntimeError(f"Failed to write STL file to {stl_path}")
        reporter.print(f"Saved STL file to {stl_path}")

        # prepare script for FreeCAD
        reporter.print(f"Converting to BRep via FreeCAD to {args.output_path} ...")
        
        # attempt FreeCAD conversion

        # assign environment variables for FreeCAD script
        env = os.environ.copy()
        env['INPUT_STL'] = os.path.abspath(stl_path)
        env['OUTPUT_BREP'] = os.path.abspath(args.output_path)
        
        # call FreeCAD in headless mode with the conversion script
        try:
            process = subprocess.run([args.freecad_cmd, args.freecad_script], env=env)
            if process.returncode != 0:
                reporter.print("FreeCAD process failed with return code:", process.returncode)
            else:
                reporter.print("FreeCAD conversion completed successfully.")
        except Exception as e:
            reporter.print("Error running FreeCAD command:", e)
    # warn if total shrinkage is above tolerance
    surface_shrinkage = abs(post_area - pre_area) / pre_area
    volume_shrinkage = abs(post_volume - pre_volume) / pre_volume
    warned = False
    if surface_shrinkage > SHRINKAGE_TOLERANCE or volume_shrinkage > SHRINKAGE_TOLERANCE:
        reporter.print("WARNING: Significant shrinkage detected!")
        reporter.print(f"Surface area shrinkage: {surface_shrinkage*100:.2f}%")
        reporter.print(f"Volume shrinkage: {volume_shrinkage*100:.2f}%")
        reporter.print("Consider adjusting smoothing or decimation parameters.")
        warned = True
    if not warned:
        reporter.print(f"Surface area shrinkage: {surface_shrinkage*100:.2f}%")
        reporter.print(f"Volume shrinkage: {volume_shrinkage*100:.2f}%")
    
    reporter.end_log()
    
    if args.open_gui:
        open3d.visualization.draw_geometries([mesh], point_show_normal=True, mesh_show_wireframe=True)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
