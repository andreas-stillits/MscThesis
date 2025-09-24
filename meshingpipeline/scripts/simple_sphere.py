""" 
simple_sphere.py

Script for automating .brep --> .msh conversion for a simple sphere geometry.
Creates a spherical airspace inside a cylinder tissue block by boolean cutting a sphere from a cylinder.

Usage:
    python simple_sphere.py <input.brep> [options]

Options:
    --suppress               Suppress verbose output
    --suppress-gmsh          Suppress Gmsh terminal output
    --output-path <path>     Path to output .msh file. if not provided, use input path with .msh extension
    --bm <float>             Boundary margin fraction for cylinder plug no-flux boundaries (default: 0.01)
    --scm <float>            Substomatal cavity margin fraction for cylinder plug (default: 0.2)
    --tolerance <float>      Tolerance for geometric comparisons (default: 0.01)
    --open-gui               Open the Gmsh GUI to visualize the mesh after generation
"""

import os
import argparse
import numpy as np 
import gmsh 

# set namespace
kernel = gmsh.model.occ

BOUNDARY_MARGIN_FRACTION = 0.01
SUBSTOMATAL_CAVITY_MARGIN_FRACTION = 0.2
TOLERANCE = 0.01

def iterative_affine_transformation(entity, transformation, error, max_iterations=5, tolerance=1e-6, target_size=1.0):
    """ 
    Iteratively apply an affine transformation to an entity until the error is below the tolerance
    """
    count = 0
    for _ in range(max_iterations):
        center, size = get_bbox(entity)
        current_error = abs(error(center, size, target_size))   
        if current_error < tolerance:
            break        
        transform = transformation(center, size, target_size)
        kernel.affineTransform(entity, transform)
        kernel.synchronize()
        count += 1
    return count


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
    p.add_argument("--tolerance", type=float, default=TOLERANCE, help=f"Tolerance for geometric comparisons (default: {TOLERANCE:.3f})")
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
    elif not args.output_path.lower().endswith(".msh"):
        raise ValueError(f"Output file {args.output_path} is not a .msh file")

    # initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) # fix getBoundingBox to use STL bounds (more robust)
    gmsh.model.add("Leaf Plug Model")

    # suppress gmsh output if requested
    if args.suppress or args.suppress_gmsh:
        gmsh.option.setNumber("General.Terminal", 0)

    # import the BRep file
    tissue = kernel.importShapes(args.input_brep) # usually [(3, 1)]
    kernel.synchronize()
    if not args.suppress:
        print(f"Imported shape from {args.input_brep}")


    # calculate cylinder geometry
    center, size   = get_bbox(tissue)
    bottom_z       = center[2] - size[2]*(0.5 + args.scm) # z-coordinate of the bottom cylinder surface
    height         = size[2]*(1 + args.scm + args.bm)
    # determine the appropriate dimensions for the cylinder plug
    bottom_surface = (center[0], center[1], bottom_z)
    axis           = (0, 0, height)
    radius         = (1+args.bm)*np.sqrt((size[0]/2)**2 + (size[1]/2)**2)
    
    if not args.suppress:
        print(f"Calculated cylinder plug dimensions:")
        print(f"  bottom_surface: {bottom_surface}")
        print(f"  axis:           {axis}")
        print(f"  radius:         {radius:.3f}")

    # create the cylinder plug
    cylinder = [(3, kernel.addCylinder(
        *bottom_surface,
        *axis,
        radius))]
    kernel.synchronize()
    
    # perform boolean cut to create airspace
    airspace, _ = kernel.cut(cylinder, tissue, removeObject=True, removeTool=True)
    kernel.synchronize()
    if not args.suppress:
        print(f"Created airspace by boolean cut of cylinder and tissue", airspace)

    # Iteratively apply affine transformation to airspace to center bottom surface at origin and scale height to 1
    transformation = lambda center, size, target_size: [
        (target_size / size[2]), 0, 0, - center[0]           *(target_size / size[2]),
        0, (target_size / size[2]), 0, - center[1]           *(target_size / size[2]),
        0, 0, (target_size / size[2]), -(center[2]-size[2]/2)*(target_size / size[2]),
        0, 0, 0, 1
    ]
    error      = lambda center, size, target_size: (size[2] - target_size)/target_size
    iterations = iterative_affine_transformation(airspace, transformation, error, max_iterations=5, target_size=1.0)

    if not args.suppress: 
        center, size = get_bbox(airspace)
        print(f"Applied {iterations} affine transformations to airspace. New center: {center}, New size: {size}")
        print(f"Finished transformations. Assigning physical groups...")
    #____________________________________________________
    
    
    # Assign Physical Groups

    # determine curved face tag 
    # OBS: this approach of identification by area only works if the curved area 2 pi r is unique up to tolerace
    # However, top and bottom surfaces will always be caught by the COM z-coordinate check below
    center, size = get_bbox(airspace)
    a = size[0]/2
    b = size[1]/2
    curved_area_target = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b))) # approximation of ellipse circumference to account for slight transform assymetry
    if not args.suppress: print("Curved area:", curved_area_target)

    curved_area_found = []
    curved_area_tag = None
    top_area_tag = None 
    bottom_area_tag = None
    def iscurved(tag):
        area = kernel.getMass(2, tag)
        trigger = abs(area/curved_area_target - 1) <= args.tolerance
        if trigger: 
            curved_area_found.append(area)
            print(f"Curved area found: {area}, relative error: {area/curved_area_target - 1:.6f}")
        return trigger
    
    # airspace
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in airspace], 1, name="airspace")
    # surfaces
    surfaces = gmsh.model.getEntities(dim=2)
    if not args.suppress: print(len(surfaces), "surfaces found")
    mesophyll_surface_tags = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0):
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
            top_area_tag = tag
            if not args.suppress: print(f"Top surface assigned to physical group 'top_surface'")
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
            bottom_area_tag = tag
            if not args.suppress: print(f"Bottom surface assigned to physical group 'bottom_surface'")
        elif iscurved(tag):
            # curved surface of cylinder
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
            curved_area_tag = tag
            if not args.suppress: print(f"Curved surface assigned to physical group 'curved_surface'")
        else:
            # other surfaces
            mesophyll_surface_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, mesophyll_surface_tags, 5, name="mesophyll_surfaces")
    if not args.suppress: print(f"Other surfaces assigned to physical group 'mesophyll_surfaces'")
    assert len(curved_area_found) == 1, f"Error identifying curved face of cylinder. Found {len(curved_area_found)} curved faces with relative errors from target: {[area/curved_area_target - 1 for area in curved_area_found]}"
    if not args.suppress: print("Physical groups assigned. Proceeding to mesh generation...")
    #____________________________________________________
    # Specify mesh size fields
    mesophyll_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(mesophyll_distance, "FacesList", mesophyll_surface_tags)
    resolution = 0.02
    minimum = 0.05 
    max     = 0.2
    mesophyll_threshold = gmsh.model.mesh.field.add("Threshold")    
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "IField", mesophyll_distance)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "LcMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "DistMin", minimum)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "DistMax", max)
    #
    inlet_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_distance, "FacesList", [bottom_area_tag])
    inlet_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_threshold, "IField", inlet_distance)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMin", 2*resolution)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMin", minimum)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMax", max)
    #
    minimum_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", [mesophyll_threshold, inlet_threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)
    kernel.synchronize()
    #____________________________________________________
    gmsh.model.mesh.generate(3)
    gmsh.write(args.output_path)
    if not args.suppress: 
        print(f"Mesh written to {args.output_path}")
    if args.open_gui: gmsh.fltk.run()

    gmsh.finalize()
    if not args.suppress: print("Gmsh finalized")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

