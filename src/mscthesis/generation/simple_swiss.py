"""  
simple_swiss.py 

Generate a simple swiss cheese mesh of non-overlapping spheres in .npy format

Usage:
    python simple_swiss.py <output_path> [options]

Options:
    --seed <int>             Random seed for sphere placement (default: 121)
"""

import argparse 
import numpy as np 
import numpy.random as r
from mscthesis.utilities.reporter import Reporter
import mscthesis.utilities.inspecter as inspecter

SEED = 121
RESOLUTION = 64
N_SPHERES = 10
MIN_RADIUS = 0.05
MAX_RADIUS = 0.15 # relative to a plug radius of 1.0
DEPTH = 2.0 # relative to a plug radius of 1.0
ALLOWED_ATTEMPTS = 1000
MIN_SPACING = 0.02



def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a simple swiss cheese mesh of non-overlapping spheres in .npy format")
    p.add_argument("output_path", type=str, default="test.npy", help="Output path for the generated geometry (.npy format)")
    p.add_argument("--seed", type=int, default=SEED, help=f"Random seed for sphere placement (default: {SEED})")
    p.add_argument("--resolution", type=int, default=RESOLUTION, help=f"Resolution of the output grid (default: {RESOLUTION})")
    p.add_argument("--n-spheres", type=int, default=N_SPHERES, help=f"Number of spheres to generate (default: {N_SPHERES})")
    p.add_argument("--min-radius", type=float, default=MIN_RADIUS, help=f"Minimum radius of spheres (default: {MIN_RADIUS})")
    p.add_argument("--max-radius", type=float, default=MAX_RADIUS, help=f"Maximum radius of spheres (default: {MAX_RADIUS})")
    p.add_argument("--depth", type=float, default=DEPTH, help=f"Depth of plug relative to plug radius of 1.0 (default: {DEPTH})")
    p.add_argument("--allowed-attempts", type=int, default=ALLOWED_ATTEMPTS, help=f"Maximum number of attempts to place spheres (default: {ALLOWED_ATTEMPTS})")
    p.add_argument("--min-spacing", type=float, default=MIN_SPACING, help=f"Minimum spacing between spheres (default: {MIN_SPACING})")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress console output")
    p.add_argument("--plot", default=False, action="store_true", help="Plot the generated geometry with open3d")
    args = p.parse_args(argv)

    # check output_path has .npy extension
    if not args.output_path.lower().endswith(".npy"):
        raise ValueError(f"Output path {args.output_path} does not have .npy extension")

    # initialize 
    reporter = Reporter(args, __file__)
    reporter.start_log()
    r.seed(args.seed)

    reporter.print(f"Generating swiss cheese geometry with approximately {args.n_spheres} spheres...  ")
    x = np.linspace(-1.0, 1.0, args.resolution)
    y = np.linspace(-1.0, 1.0, args.resolution)
    z = np.linspace(0, args.depth, args.resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    voxels = np.zeros((args.resolution, args.resolution, args.resolution), dtype=np.uint8)
    centers = []
    radii = []

    max_xy = 1.0 - args.max_radius - args.min_spacing
    min_z  = args.max_radius + args.min_spacing
    max_z  = args.depth - args.max_radius - args.min_spacing

    reporter.print("Attempting to place spheres...  ")
    for _ in range(args.n_spheres):
        attempts = 0
        while attempts < args.allowed_attempts:
            center = np.array([r.uniform(-max_xy, max_xy), 
                      r.uniform(-max_xy, max_xy), 
                      r.uniform(min_z, max_z)])
            radius = r.uniform(args.min_radius, args.max_radius)
            if all(np.linalg.norm(center - cen) > (radius + rad + args.min_spacing) for cen, rad in zip(centers, radii)) and np.linalg.norm(center[:2]) <= max_xy:
                centers.append(center)
                radii.append(radius)
                break
            attempts += 1
        
        else: # executed if while loop is not stopped by break - then we dont attempt to place any further spheres
            break # break out of the for loop
        
        distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        voxels |= (distance <= radius).astype(np.uint8)

    if len(centers) < args.n_spheres:
        reporter.print(f"WARNING: Only packed {len(centers)} spheres out of {args.n_spheres} after {args.allowed_attempts} attempts per sphere. Consider increasing --allowed-attempts or reducing --n-spheres.")
    else:
        reporter.print(f"Successfully packed {args.n_spheres} spheres.")    

    reporter.print(f"Saving geometry to {args.output_path}...  ")
    np.save(args.output_path, voxels)

    reporter.end_log()

    # plot using the inspecter utility
    if args.plot:
        inspecter.main([args.output_path])

    return 0



if __name__ == "__main__":
    raise SystemExit(main())