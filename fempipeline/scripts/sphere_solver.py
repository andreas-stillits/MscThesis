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
from dolfinx import default_scalar_type
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner
import numpy as np



def main(argv=None):
    p = argparse.ArgumentParser(description="Solve a PDE on a sphere mesh using FEniCSx.")
    p.add_argument("input_msh", type=str, help="Input mesh file in .msh format")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress output messages")
    p.add_argument("--order", type=int, default=1, help="Finite element order (default: 1)")
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


    return 0



if __name__ == "__main__":
    raise SystemExit(main())


