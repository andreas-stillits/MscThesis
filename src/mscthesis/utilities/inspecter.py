""" 
quick_plot.py 

Functionality to quickly visualize mesh-related data files (.npy, .stl, .msh (3D), .msh (with physical groups))
The appropriate visualization tool is chosen based on file extension.

Usage:
    python quick_plot.py <input_path> [--groups]

    --groups: If input is a .msh file, plot physical groups

"""
import os
import argparse 
import numpy as np
import gmsh
import pyvista as pv 
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
from mpi4py import MPI 
import adios4dolfinx as a4x

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import open3d

def inspecter(args: argparse.Namespace):
    """ Quick plot of a datafile """
    # check if file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file {args.input_path} not found")
    
    # Branch according to file extension
    ext = os.path.splitext(args.input_path)[1].lower()
    #
    if ext == ".npy":
        voxels = np.load(args.input_path)
        points = np.argwhere(voxels > 0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        open3d.visualization.draw_geometries([pcd])
    #
    elif ext == ".stl":
        mesh = open3d.io.read_triangle_mesh(args.input_path)
        mesh.compute_vertex_normals()  # Compute normals if not present
        open3d.visualization.draw_geometries([mesh], point_show_normal=True, mesh_show_wireframe=True)
    #
    elif ext == ".msh":
        if args.groups:
            rank = 0
            mesh, cell_tags, facet_tags = gmshio.read_from_msh(args.input_path, MPI.COMM_WORLD, rank, gdim=3)
            topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim-1)

            # visualize with pyvista
            if MPI.COMM_WORLD.rank == 0:
                grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                facet_values = facet_tags.values 
                facet_indices = facet_tags.indices
                facet_marker_array = np.full(grid.n_cells, np.nan)
                for i, idx in enumerate(facet_indices):
                    facet_marker_array[idx] = facet_values[i]
                grid.cell_data["marker"] = facet_marker_array    
                pl = pv.Plotter()
                pl.add_mesh(grid, show_edges=True, scalars="marker", cmap="tab10", nan_opacity=0, 
                            scalar_bar_args={"title": "Physical groups", "vertical": True})
                pl.show_axes()
                pl.show()
        else:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("Mesh from file")
            gmsh.merge(args.input_path)
            gmsh.fltk.run()
            gmsh.finalize()
    elif ext == ".bp":
        mesh = a4x.read_mesh(args.input_path, MPI.COMM_WORLD)
        cell_tags = a4x.read_meshtags(args.input_path, mesh, meshtag_name="cell_tags")
        facet_tags = a4x.read_meshtags(args.input_path, mesh, meshtag_name="facet_tags")
        #
        V = fem.functionspace(mesh, ("Lagrange", 1))
        uh = fem.Function(V)
        a4x.read_function(args.input_path, uh, name="solution")
        #        
        if MPI.COMM_WORLD.rank == 0:
            topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            grid.point_data["uh"] = uh.x.array.real

            xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
            slices = grid.slice_orthogonal(x=(xmin+xmax)/2, y=(ymin+ymax)/2, z=(zmin+zmax)/2)
            p = pv.Plotter()
            p.add_mesh(slices, scalars="uh", cmap="viridis", clim=[0, np.max(uh.x.array.real)])
            p.add_mesh(grid.outline(), color="k")
            p.show_axes()
            p.show()

def parse_args(argv=None):
    """ Parse command line arguments """
    p = argparse.ArgumentParser(description="Quick plot of a mesh-related datafile")
    p.add_argument("input_path", type=str, help="Path to input mesh file (.npy, .stl, .msh)")
    p.add_argument("--groups", default=False, action="store_true", help="If input is a .msh file, plot physical groups")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inspecter(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())