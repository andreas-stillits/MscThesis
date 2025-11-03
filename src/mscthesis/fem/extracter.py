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
# OBS


# default parameters
DEFAULT_RESOLUTION = 50
TOLERANCE = 1e-3


def main(argv=None):
    p = argparse.ArgumentParser(description="Functionality to extract the xy mean and variance of CO2 field as a function of z")
    p.add_argument("input_path", type=str, help="Input path must be a .bp folder")
    p.add_argument("--output_path", type=str, help="Output path must be a .txt file")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help=f"Resolution along z-axis (default: {DEFAULT_RESOLUTION})")
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
    tdim = mesh.topology.dim
    print(f"DEBUG: zmin={zmin},\n zmax={zmax},\n dz={dz},\n edges={edges},\n centers={centers} \n _____\n")

    # cell midpoints (vectorized)
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    cell2vert = mesh.topology.connectivity(tdim, 0)
    x = mesh.geometry.x
    cell_mid_z = np.array([x[cell2vert.links(c)].mean(axis=0)[2] for c in cells])
    print(f"DEBUG: cell_mid_z = {cell_mid_z} \n _____\n")

    # Create MeshTags for cells: tag each cell by its slice index k
    cell_tags = np.full(cells.shape, -1, dtype=np.int32)
    for k, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (cell_mid_z >= a) & (cell_mid_z < b if k < len(centers)-1 else cell_mid_z <= b)
        cell_tags[m] = k

    ct = dmesh.meshtags(mesh, tdim, cells, cell_tags)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

    # V = uh.function_space  # your solved solution uh \in V

    N = len(centers)
    zs = np.zeros(N)
    V_solids = np.zeros(N)
    u_means  = np.zeros(N)
    u2_means = np.zeros(N)

    for k in range(len(centers)):
        # Solid volume in slice k
        V_solid = fem.assemble_scalar(fem.form(1 * dx(k)))
        # Average u over solid in slice
        U_int = fem.assemble_scalar(fem.form(uh * dx(k)))
        u_avg = U_int / V_solid if V_solid > 0 else np.nan
        # Average u^2 over solid in slice
        U2_int = fem.assemble_scalar(fem.form(uh**2 * dx(k)))
        u2_avg = U2_int / V_solid if V_solid > 0 else np.nan
        # Append slice statistics
        zs[k] = centers[k]
        V_solids[k] = V_solid
        u_means[k] = u_avg
        u2_means[k] = u2_avg
    
    u_std = np.sqrt(u2_means - u_means**2)

    data = np.vstack([zs, V_solids, u_means, u2_means, u_std]).T
    np.savetxt(args.output_path, data, header="z V_solid u_mean u2_mean u_std", delimiter=";")
    reporter.print(f"Saved extracted data to {args.output_path}.")

    # optional plotting
    if args.plot:
        reporter.print("Plotting mean and variance profiles...")
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('z')
        ax1.set_ylabel('Mean CO2', color=color)
        ax1.plot(zs, u_means, color=color, label='Mean CO2')
        ax1.tick_params(axis='y', labelcolor=color)

        # plot std as a fill between band around the mean
        ax1.fill_between(zs, u_means - u_std, u_means + u_std, color=color, alpha=0.3, label='Std Dev Band')

        # plot v_solid on secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Solid Volume', color=color)
        ax2.plot(zs, V_solids, color=color, linestyle='--', label='Solid Volume')
        ax2.tick_params(axis='y', labelcolor=color)
    
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.1*np.max(V_solids))
        ax1.grid()

        fig.tight_layout()  
        plt.legend()
        plt.title("CO2 Mean and Standard Deviation Profiles along z-axis")
        plt.show()
        reporter.print("Plotting completed.")


    reporter.end_log()
    
    return 0



if __name__ == "__main__":
    raise SystemExit(main())