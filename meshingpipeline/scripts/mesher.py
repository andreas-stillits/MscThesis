""" 
mesher.py
Generates a volumentric mesh from a BRep file using Gmsh's Python API.
Creates a bounding box around the shape, centers and scales it to unit height
Assigns physical groups and saves the mesh in MSH format.

Usage:
    python mesher.py input.brep [options]

Options:
    --suppress               Suppress verbose output
    --suppress-gmsh          Suppress Gmsh terminal output (default True)
    --paradermal-margin f    Margin fraction for the free bbox space in the paradermal direction (default 0.05)
    --substomatal-margin f   Margin fraction for the free bbox space in the substomatal cavity direction (default 0.1)
    --open-gui               Open the Gmsh GUI to visualize the mesh after generation

"""

import argparse
import os
import numpy as np 
import gmsh 

# set namespace
kernel = gmsh.model.occ

PARADERMAL_MARGIN_FRACTION = 0.05
SUBSTOMATAL_CAVITY_MARGIN_FRACTION = 0.1


def get_bbox(dim, tag):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    return bbox_center, bbox_size


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate volumetric mesh from BRep file using Gmsh.")
    p.add_argument("input_brep", type=str, help="Input BRep file.")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress verbose output")
    p.add_argument("--suppress-gmsh", default=True, action="store_false", help="Suppress Gmsh terminal output")
    p.add_argument("--paradermal-margin", type=float, default=PARADERMAL_MARGIN_FRACTION, help=f"Margin fraction for the free bbox space in the paradermal direction (default {PARADERMAL_MARGIN_FRACTION})")
    p.add_argument("--substomatal-margin", type=float, default=SUBSTOMATAL_CAVITY_MARGIN_FRACTION, help=f"Margin fraction for the free bbox space in the substomatal cavity direction (default {SUBSTOMATAL_CAVITY_MARGIN_FRACTION})")
    p.add_argument("--open-gui", default=False, action="store_true", help="Open the Gmsh GUI to visualize the mesh after generation")
    #
    args = p.parse_args(argv)

    # check if file exists
    if not os.path.isfile(args.input_brep):
        raise FileNotFoundError(f"Input file {args.input_brep} does not exist.")
    
    # initialize gmsh
    gmsh.initialize()
    gmsh.model.add("Leaf_plug_model")

    # suppress gmsh output if requested
    if args.suppress or args.suppress_gmsh:
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output

    # import the BRep file
    shape = kernel.importShapes(args.input_brep)
    kernel.synchronize()
    if not args.suppress:
        print(f"Imported shape from {args.input_brep}")

    # center and scale to unit height
    center, size = get_bbox(*shape[0])
    scale = 1.0 / size[2]  # size[2] is the z-extent
    kernel.translate(shape, -center[0], -center[1], -(center[2]-size[2]/2))
    kernel.dilate(shape, 0, 0, 0, scale, scale, scale)
    kernel.translate(shape, 0, 0, args.substomatal_margin) # has unit height now so no multiplication needed
    kernel.synchronize()

    # draw a box matching the shape's bounding box (plus some margin)
    center, size = get_bbox(*shape[0])
    marginx = args.paradermal_margin * size[0] / 2
    marginy = args.paradermal_margin * size[1] / 2
    marginz = args.substomatal_margin * size[2]
    box = kernel.addBox(center[0] - size[0]/2 - marginx, center[1] - size[1]/2 - marginy, 0,
                  size[0] + 2*marginx, size[1] + 2*marginy, size[2] + marginz)
    kernel.synchronize()
    # Perform boolean difference with the extended box and the imported shape
    volumes = gmsh.model.getEntities(dim=3)
    cells = [vol[1] for vol in volumes if vol[1] != box]
    airspace, outDimTagMap = kernel.cut([(3, box)], [(3, cell) for cell in cells])
    # scale the airspace to unit height
    scale = 1.0 / (1 + marginz)
    kernel.dilate(airspace, 0, 0, 0, scale, scale, scale)
    kernel.synchronize()

    # generate mesh
    gmsh.model.mesh.generate(3)
    if args.open_gui:
        gmsh.fltk.run()  # Open the GUI to visualize the mesh

    # save mesh
    output_mesh = os.path.splitext(args.input_brep)[0] + ".msh"
    gmsh.write(output_mesh)
    # finalize gmsh
    gmsh.finalize()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())