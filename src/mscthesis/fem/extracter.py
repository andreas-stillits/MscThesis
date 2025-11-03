"""
extracter.py

Functionality to extract the xy mean and variance of CO2 field as a function of z

Usage:
    python extracter.py input_path.bp [options]

Options:
    --resolution <int>         Resolution along z-axis (default: 100)

"""

import argparse
from mscthesis.utilities.reporter import Reporter
from mscthesis.utilities.checker import check_io_paths
from mpi4py import MPI
import adios4dolfinx as a4x
from dolfinx import fem, mesh as dmesh
import ufl
import matplotlib.pyplot as plt 
import numpy as np 

# OBS
#
# OBS:LATER ON, EXTRACTION LIKE THIS IS HIGHLY PARALLELIZABLE!
#
# NB: IMPLEMENT "inside" AS CONTINUOUS tanh INSTEAD OF CONDITIONAL TO GET A SMOOTHER BEHAVIOUR
#
# NB: CONSIDER ANNOTATING PHYSICAL GROUPS CORRESPONDING TO z-SLICES IN GMSH. *dont know if it is less stable/efficient that quadrature-based extraction though*
#
# OBS


# default parameters
DEFAULT_RESOLUTION = 50
DEFAULT_QDEGREE = 8
TOLERANCE = 1e-3


def main(argv=None):
    p = argparse.ArgumentParser(description="Functionality to extract the xy mean and variance of CO2 field as a function of z")
    p.add_argument("input_path", type=str, help="Input path must be a .bp folder")
    p.add_argument("--output_path", type=str, help="Output path must be a .txt file")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help=f"Resolution along z-axis (default: {DEFAULT_RESOLUTION})")
    p.add_argument("--qdegree", type=int, default=DEFAULT_QDEGREE, help=f"Quadrature degree for integration (default: {DEFAULT_QDEGREE})")
    p.add_argument("--plot", default=False, action="store_true", help="If set, plot the mean and variance profiles")
    args = p.parse_args(argv)

    check_io_paths(args, ".bp", ".txt") # input and output extensions
    reporter = Reporter(args, __file__)
    reporter.start_log()
    
    # load data
    reporter.print(f"Loading data from {args.input_path}...")
    mesh = a4x.read_mesh(args.input_path, MPI.COMM_WORLD)
    #
    V = fem.functionspace(mesh, ("Lagrange", 1))
    uh = fem.Function(V)
    a4x.read_function(args.input_path, uh, name="solution")

    # create z bins and compute first and second moments across xy slices 
    reporter.print("Extracting mean and variance profiles along z-axis...")
    zmin, zmax = mesh.geometry.x[:, 2].min(), mesh.geometry.x[:, 2].max()
    dz = (zmax - zmin) / args.resolution
    edges = np.arange(zmin, zmax + TOLERANCE, dz)
    centers = (edges[:-1] + edges[1:]) / 2

    z = ufl.SpatialCoordinate(mesh)[2]

    def get_slice_quantities(a, b, qdeg=DEFAULT_QDEGREE):
        inside = ufl.conditional(ufl.And(ufl.ge(z, a), ufl.lt(z, b)), 1.0, 0.0)
        dx_q = ufl.dx(metadata={"quadrature_degree": qdeg})
        V_solid = fem.assemble_scalar(fem.form(inside * dx_q))
        U_int = fem.assemble_scalar(fem.form(uh * inside * dx_q))
        U2_int = fem.assemble_scalar(fem.form(uh**2 * inside * dx_q))
        u_avg = U_int / V_solid if V_solid > 0 else np.nan
        u2_avg = U2_int / V_solid if V_solid > 0 else np.nan
        return V_solid, u_avg, u2_avg

    quantities = np.vstack([list(get_slice_quantities(a, b, qdeg=args.qdegree)) for a, b in zip(edges[:-1], edges[1:])])

    V_solids = quantities[:, 0]
    u_means  = quantities[:, 1]
    u2_means = quantities[:, 2]
    u_std = np.sqrt(u2_means - u_means**2)


    data = np.vstack([centers, V_solids, u_means, u2_means, u_std]).T
    np.savetxt(args.output_path, data, header="z V_solid u_mean u2_mean u_std", delimiter=";")
    reporter.print(f"Saved extracted data to {args.output_path}.")
    reporter.end_log()

    # optional plotting
    if args.plot:
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('z')
        ax1.set_ylabel('Mean CO2', color=color)
        ax1.plot(centers, u_means, color=color, label='Mean CO2')
        ax1.tick_params(axis='y', labelcolor=color)

        # plot std as a fill between band around the mean
        ax1.fill_between(centers, u_means - u_std, u_means + u_std, color=color, alpha=0.3, label='Std Dev Band')

        # plot v_solid on secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Solid Volume', color=color)
        ax2.plot(centers, V_solids, color=color, linestyle='--', label='Solid Volume')
        ax2.tick_params(axis='y', labelcolor=color)
    
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.1*np.max(V_solids))
        ax1.grid()

        fig.tight_layout()  
        plt.legend()
        plt.title("CO2 Mean and Standard Deviation Profiles along z-axis")
        plt.show()
    
    return 0



if __name__ == "__main__":
    raise SystemExit(main())