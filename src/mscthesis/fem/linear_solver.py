"""  
sphere_solver.py

Solve a PDE on a sphere mesh using FEniCSx.

Usage:
    python sphere_solver.py <mesh_file.msh> [options]

Options:
    --suppress              Suppress output messages
    --order <int>           Finite element order (default: 1)
    --output-path <str>     Output file for solution
    --plot                  Plot the solution using pyvista

"""

import os
import argparse
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np 
import pyvista as pv

import ufl 
from dolfinx import fem, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio, XDMFFile 
from mscthesis.utilities.reporter import Reporter

ORDER = 1 # Default finite element order
DIFFUSIVITY = 0.1
STOMATAL_CONDUCTANCE = 10.0
STOMATAL_RADIUS = 0.1
STOMATAL_BLUR = 0.02
REACTIVITY = 2.0
ATMOSPHERIC_CONC = 1.0
COMPENSATION_POINT = 0.1



def main(argv=None):
    p = argparse.ArgumentParser(description="Solve a PDE on a sphere mesh using FEniCSx.")
    p.add_argument("input_msh", type=str, help="Input mesh file in .msh format")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress output messages")
    p.add_argument("--order", type=int, default=ORDER, help=f"Finite element order (default: {ORDER})")
    p.add_argument("--output-path", type=str, default=None, help="Output path for solution. If not provided, will use input path with .xdmf extension.")
    p.add_argument("--plot", default=False, action="store_true", help="Plot the solution using pyvista")
    args = p.parse_args(argv)
    
    # check if input file exists
    if not os.path.isfile(args.input_msh):
        raise FileNotFoundError(f"Input file {args.input_msh} does not exist.")

    # check if .msh format
    if not args.input_msh.lower().endswith(".msh"):
        raise ValueError(f"Input file {args.input_msh} is not a .msh file.")
    
    # derive output path if not provided
    if args.output_path is None:
        args.output_path = os.path.splitext(args.input_msh)[0] + ".xdmf"
    elif not args.output_path.lower().endswith(".xdmf"):
        raise ValueError(f"Output file {args.output_path} is not a .xdmf file.")

    reporter = Reporter(args, __file__)
    reporter.start_log()

    # Load mesh
    reporter.print(f"Loading mesh from {args.input_msh}...")
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(args.input_msh, MPI.COMM_WORLD, 0, gdim=3)
    
    # tags for physical groups
    AIRSPACE_VOLUME_TAG = 1 
    TOP_SURFACE_TAG = 2
    BOTTOM_SURFACE_TAG = 3
    CURVED_SURFACE_TAG = 4
    MESOPHYLL_SURFACE_TAG = 5

    # define function space
    V = fem.functionspace(mesh, ("Lagrange", args.order))
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    reporter.print(f"Function space of order {args.order} created.")

    # create trial and test functions
    u   = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)
    x   = ufl.SpatialCoordinate(mesh)
    phi = ( x[0]**2 + x[1]**2 - STOMATAL_RADIUS**2)
    gs  = STOMATAL_CONDUCTANCE * 0.5 * (1.0 - ufl.tanh(phi/STOMATAL_BLUR))

    D  = fem.Constant(mesh, PETSc.ScalarType(DIFFUSIVITY)) 
    K  = fem.Constant(mesh, PETSc.ScalarType(REACTIVITY))
    Ca = fem.Constant(mesh, PETSc.ScalarType(ATMOSPHERIC_CONC))
    C_ = fem.Constant(mesh, PETSc.ScalarType(COMPENSATION_POINT))
    f  = fem.Constant(mesh, PETSc.ScalarType(0.0))

    # Define variational problem
    reporter.print("Setting up variational problem...")
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(AIRSPACE_VOLUME_TAG) \
        + gs/D * u * v * ds(BOTTOM_SURFACE_TAG) \
        + K/D * u * v * ds(MESOPHYLL_SURFACE_TAG)

    L = f * v * dx(AIRSPACE_VOLUME_TAG) \
        + gs/D * Ca * v * ds(BOTTOM_SURFACE_TAG) \
        + K/D * C_ * v * ds(MESOPHYLL_SURFACE_TAG)
    
    # NB: above the signs fit the normal direction as assigned by gmsh

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    reporter.print("Solving linear system...")
    uh = problem.solve() 
    reporter.print("Linear system solved.")

    # Save solution to XDMF
    reporter.print(f"Saving solution to {args.output_path}...")
    with XDMFFile(MPI.COMM_WORLD, args.output_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)

    if args.plot and MPI.COMM_WORLD.rank == 0:
        topology, cell_types, geometry = plot.vtk_mesh(mesh, mesh.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["uh"] = uh.x.array.real

        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
        slices = grid.slice_orthogonal(x=(xmin+xmax)/2, y=(ymin+ymax)/2, z=(zmin+zmax)/2)
        p = pv.Plotter()
        p.add_mesh(slices, scalars="uh", cmap="viridis", clim=[0, ATMOSPHERIC_CONC])
        p.add_mesh(grid.outline(), color="k")
        p.show_axes()
        p.show()

    reporter.close()

    return 0



if __name__ == "__main__":
    raise SystemExit(main())


