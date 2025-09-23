""" 
simple_sphere.py

docstring


"""

import os
import argparse
import numpy as np 
import gmsh 

# set namespace
kernel = gmsh.model.occ

BOUNDARY_MARGIN_FRACTION = 0.01
SUBSTOMATAL_CAVITY_MARGIN_FRACTION = 0.2
TOLERANCE = 1e-5

def iterative_affine_transformation(entity, transformation, error, max_iterations=5, tolerance=1e-6, target_size=1.0):
    """ 
    Apply iterative affine transformation to an entity until the error is below the tolerance
    """
    for iteration in range(max_iterations):
        center, size = get_bbox(entity)
        current_error = abs(error(center, size, target_size))   
        if current_error < tolerance:
            break        
        transform = transformation(center, size, target_size)
        kernel.affineTransform(entity, transform)
        kernel.synchronize()
    return iteration + 1


def get_bbox(entity):
    """
    Get bounding box of a given entity
    entity: [(dim, tag)]
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(*entity[0])
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    return bbox_center, bbox_size


def main(argv=None):
    p = argparse.ArgumentParser(description="Script for automating .brep --> .msh conversion")
    p.add_argument("input_brep", type=str, help="Path to input .brep file")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress verbose output")
    p.add_argument("--suppress-gmsh", default=True, action="store_false", help="Suppress Gmsh terminal output")
    p.add_argument("--output-path", type=str, default=None, help="Path to output .msh file. if not provided, will use input path with .msh extension")
    p.add_argument("--bm", type=float, default=BOUNDARY_MARGIN_FRACTION, help=f"Boundary margin fraction for cylinder plug no-flux boundaries (default: {BOUNDARY_MARGIN_FRACTION:.2f})")
    p.add_argument("--scm", type=float, default=SUBSTOMATAL_CAVITY_MARGIN_FRACTION, help=f"Substomatal cavity margin fraction for cylinder plug (default: {SUBSTOMATAL_CAVITY_MARGIN_FRACTION:.2f})")
    p.add_argument("--open-gui", default=False, action="store_true", help="Open the Gmsh GUI to visualize the mesh after generation")
    args = p.parse_args(argv)
    
    # check if input file exists
    if not os.path.isfile(args.input_brep):
        raise FileNotFoundError(f"Input file {args.input_brep} does not exist")

    # check if input file is .brep
    if not args.input_brep.lower().endswith(".brep"):
        raise ValueError(f"Input file {args.input_brep} is not a .brep file")

    # set output path
    if args.output_path is None:
        args.output_path = os.path.splitext(args.input_brep)[0] + ".msh"

    # initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) # fix getBoundingBox to use STL bounds (more robust)
    gmsh.model.add("Leaf plug model of simple single sphere")

    # suppress gmsh output if requested
    if args.suppress or args.suppress_gmsh:
        gmsh.option.setNumber("General.Terminal", 0)

    # import the BRep file
    tissue = kernel.importShapes(args.input_brep) # usually [(3, 1)]
    kernel.synchronize()
    if not args.suppress:
        print(f"Imported shape from {args.input_brep}")


    # get bounding box
    center, size = get_bbox(tissue)
    center_z = center[2] - size[2]*(0.5 + args.scm)
    height   = size[2]*(1 + args.scm + args.bm)

    # determine the appropriate dimensions for the cylinder plug
    center_surface = (center[0], center[1], center_z)
    axis           = (0, 0, height)
    radius         = (1+args.bm)*np.sqrt((size[0]/2)**2 + (size[1]/2)**2)
    
    # create the cylinder plug
    cylinder = [(3, kernel.addCylinder(
        *center_surface,
        *axis,
        radius))]
    kernel.synchronize()
    
    # determine curved face tag 
    outer_faces = gmsh.model.getBoundary(cylinder, False, False)
    curved_area = 2*np.pi*radius*height
    is_curved = lambda tag: abs(kernel.getMass(2, tag)/curved_area - 1) <= TOLERANCE
    curved_face = [(2, tag) for (dim, tag) in outer_faces if is_curved(tag)]
    assert len(curved_face) == 1, "Error identifying curved face of cylinder"
    print("curved area entity", curved_face)
    
    # perform boolean cut to create airspace
    airspace, outDimTagMap = kernel.cut(cylinder, tissue, removeObject=True, removeTool=True)
    kernel.synchronize()

    # transform so center_surface is at origin and height is 1
    scale = 1.0 / height
    transform = [
        scale, 0, 0, -center_surface[0]*scale,
        0, scale, 0, -center_surface[1]*scale,
        0, 0, scale, -center_surface[2]*scale,
        0, 0, 0, 1
    ]
    kernel.affineTransform(airspace, transform)
    kernel.synchronize()
    if not args.suppress: print(f"Finished transformations. Assigning physical groups...")
    #____________________________________________________
    
    
    # Assign Physical Groups
    # airspace
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in airspace], 1, name="airspace")
    # surfaces
    surfaces = gmsh.model.getEntities(dim=2)
    print(len(surfaces), "surfaces found")
    mesophyll_surface_tags = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0):
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
            if not args.suppress: print(f"Top surface assigned to physical group 'top_surface'")
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
            if not args.suppress: print(f"Bottom surface assigned to physical group 'bottom_surface'")
        elif tag == curved_face[0][1]:
            # curved surface of cylinder
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
            if not args.suppress: print(f"Curved surface assigned to physical group 'curved_surface'")
        else:
            # other surfaces
            mesophyll_surface_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, mesophyll_surface_tags, 5, name="mesophyll_surfaces")
    if not args.suppress: print(f"Other surfaces assigned to physical group 'mesophyll_surfaces'")
    #____________________________________________________

    gmsh.model.mesh.generate(3)
    gmsh.write(args.output_path)
    if not args.suppress: print(f"Mesh written to {args.output_path}")
    if args.open_gui: gmsh.fltk.run()

    gmsh.finalize()
    print("Gmsh finalized")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

