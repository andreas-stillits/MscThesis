""" 
npy_to_brep.py
Convert a voxel representation (3D numpy array .npy) to a BREP file using FreeCAD.

The FreeCAD conversion is done by writing a temporary Python script 
that FreeCAD executes in headless mode. The input filename is passed 
via the environment variable 'INPUT_NPY'.

The code will produce an .stl and a .brep file in the same directory
as the input .npy file and inherit its file name.

Usage:
    python npy_to_brep.py path/to/your_file.npy [options]

Options:
    --spacing sx,sy,sz       Voxel spacing in each dimension (default 1.0,1.0,1.0)
    --freecad-cmd PATH       Path to the FreeCAD command line tool (default: freecadcmd-daily)
    --freecad-script PATH    Path to a custom FreeCAD conversion script (default assumes script is in the same directory as this file)
    --suppress               Suppress verbose output
    --smoothing_iter N       Number of Taubin smoothing iterations (default 10)
    --decimate N             Target number of triangles after decimation (default 10,000)
"""

import os
import subprocess
import argparse
import numpy as np 
import open3d as o3d
from skimage import measure

DEFAULT_FREECAD_CMD = "freecadcmd-daily"
DEFAULT_FREECAD_SCRIPT = os.path.join(os.path.dirname(__file__), "freecad_converter.py")
DEFAULT_SMOOTHING_ITER = 10
DEFAULT_DECIMATE = 10_000


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert water tight voxel representation in .npy to CAD model in .brep")
    p.add_argument("input_npy", type=str, help="Path to the input .npy file containing a 3D array (boolean or 0/1)")
    p.add_argument("--spacing", type=lambda s: tuple(float(x) for x in s.split(',')), default=(1.0,1.0,1.0), help="voxel spacing sx,sy,sz (default 1.0,1.0,1.0)")
    p.add_argument("--freecad-cmd", default=DEFAULT_FREECAD_CMD, help="Path to the FreeCAD command line tool")
    p.add_argument("--freecad-script", default=DEFAULT_FREECAD_SCRIPT, help=f"Path to a custom FreeCAD conversion script (default assumes script is in the same directory as {__file__})") 
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress verbose output")
    p.add_argument("--smoothing_iter", type=int, default=DEFAULT_SMOOTHING_ITER, help="Number of Taubin smoothing iterations (default 10)")
    p.add_argument("--decimate", type=int, default=DEFAULT_DECIMATE, help="Target number of triangles after decimation (default 10.000)")
    args = p.parse_args(argv)
    
    # make sure that the input file exists
    if not os.path.isfile(args.input_npy):
        raise FileNotFoundError(f"Input file {args.input_npy} does not exist.")
    
    # load file
    voxels = np.load(args.input_npy)
    if not args.suppress: print("loaded volume shape:", voxels.shape, 'spacing:', args.spacing)

    # convert to triangular surface mesh using Open3D
    verts, faces, normals, values = measure.marching_cubes(voxels, spacing=args.spacing, level=0.5)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh = clean_mesh(mesh)
    if not args.suppress: print("Initial triangles=", len(np.asarray(mesh.triangles)), "vertices=", len(np.asarray(mesh.vertices)))

    # apply smoothing
    if not args.suppress: print(f"Applying {args.smoothing_iter} Taubin smoothing iterations...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smoothing_iter)
    mesh = clean_mesh(mesh)

    # apply mesh decimation
    if not args.suppress: print(f"Decimating mesh to ~{args.decimate} triangles...")
    current = len(np.asarray(mesh.triangles)) # get current triangle count
    target = args.decimate
    if not target >= current:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    mesh = clean_mesh(mesh)
    if not args.suppress: print("Post-decimation triangles=", len(np.asarray(mesh.triangles)), "vertices=", len(np.asarray(mesh.vertices)))

    # check that mesh is manifold and water tight
    edge_manifold = mesh.is_edge_manifold()
    vertex_manifold = mesh.is_vertex_manifold()
    watertight = mesh.is_watertight()
    if not args.suppress: print("Mesh manifold:", edge_manifold, vertex_manifold, "Watertight:", watertight)
    
    # abort if these conditions are not met
    if not (edge_manifold and vertex_manifold and watertight):
        print("Error: Mesh is not manifold and watertight. Cannot convert to BREP.")
        # save as .stl for inspection
        stl_path = os.path.splitext(args.input_npy)[0] + ".stl"
        written = o3d.io.write_triangle_mesh(stl_path, mesh)
        if not written:
            raise RuntimeError(f"Failed to write STL file to {stl_path}")
        print(f"Saved non-manifold mesh as {stl_path} for inspection.")
        
    else:
        # given that the mesh is manifold and water tight, save as .stl and proceed to BREP conversion
        stl_path = os.path.splitext(args.input_npy)[0] + ".stl"
        written = o3d.io.write_triangle_mesh(stl_path, mesh)
        if not written:
            raise RuntimeError(f"Failed to write STL file to {stl_path}")
        if not args.suppress: print(f"Saved STL file to {stl_path}")

        # prepare script for FreeCAD
        brep_path = os.path.splitext(args.input_npy)[0] + ".brep"
        if not args.suppress: print(f"Converting to BRep via FreeCAD to {brep_path} ...")
        
        # attempt FreeCAD conversion

        # assign environment variables for FreeCAD script
        env = os.environ.copy()
        env['INPUT_STL'] = os.path.abspath(stl_path)
        env['OUTPUT_BREP'] = os.path.abspath(brep_path)
        
        # call FreeCAD in headless mode with the conversion script
        try:
            process = subprocess.run([args.freecad_cmd, args.freecad_script], env=env)
            if process.returncode != 0:
                print("FreeCAD process failed with return code:", process.returncode)
            else:
                if not args.suppress: print("FreeCAD conversion completed successfully.")
        except Exception as e:
            print("Error running FreeCAD command:", e)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
