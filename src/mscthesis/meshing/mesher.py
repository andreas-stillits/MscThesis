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
from mscthesis.utilities.reporter import Reporter
from mscthesis.utilities.checker import check_io_paths

# set namespace
kernel = gmsh.model.occ

BOUNDARY_MARGIN_FRACTION = 0.05
SUBSTOMATAL_CAVITY_MARGIN_FRACTION = 0.2
TOLERANCE = 0.01
MESH_SCALE = 1.0
MINIMUM_RESOLUTION = 0.02 * MESH_SCALE
MAXIMUM_RESOLUTION = 0.2 * MESH_SCALE
MINIMUM_DISTANCE   = 0.05
MAXIMUM_DISTANCE   = 0.2
INLET_BASE_RESOLUTION_FACTOR = 2.0

def iterative_affine_transformation(entity: list[tuple[int, int]], transformation: callable, error: callable, max_iterations: int = 5, tolerance: float = 1e-6, target_size: float =1.0):
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


def get_bbox(entity: list[tuple[int, int]]):
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
    p.add_argument("input_path", type=str, help="Path to input .brep file")
    p.add_argument("--output-path", type=str, default=None, help="Path to output .msh file. if not provided, will use input path with .msh extension")
    p.add_argument("--suppress", default=False, action="store_true", help="Suppress verbose output")
    p.add_argument("--suppress-gmsh", default=True, action="store_false", help="Suppress Gmsh terminal output")   
    p.add_argument("--bm", type=float, default=BOUNDARY_MARGIN_FRACTION, help=f"Boundary margin fraction for cylinder plug no-flux boundaries (default: {BOUNDARY_MARGIN_FRACTION:.2f})")
    p.add_argument("--scm", type=float, default=SUBSTOMATAL_CAVITY_MARGIN_FRACTION, help=f"Substomatal cavity margin fraction for cylinder plug (default: {SUBSTOMATAL_CAVITY_MARGIN_FRACTION:.2f})")
    p.add_argument("--tolerance", type=float, default=TOLERANCE, help=f"Tolerance for geometric comparisons (default: {TOLERANCE:.3f})")
    p.add_argument("--mesh-scale", type=float, default=MESH_SCALE, help=f"Global mesh scale factor (default: {MESH_SCALE:.2f})")
    p.add_argument("--min-resolution", type=float, default=MINIMUM_RESOLUTION, help=f"Minimum mesh resolution (default: {MINIMUM_RESOLUTION:.3f})")
    p.add_argument("--max-resolution", type=float, default=MAXIMUM_RESOLUTION, help=f"Maximum mesh resolution (default: {MAXIMUM_RESOLUTION:.3f})")
    p.add_argument("--min-distance", type=float, default=MINIMUM_DISTANCE, help=f"Minimum distance for mesh size field (default: {MINIMUM_DISTANCE:.3f})")
    p.add_argument("--max-distance", type=float, default=MAXIMUM_DISTANCE, help=f"Maximum distance for mesh size field (default: {MAXIMUM_DISTANCE:.3f})")
    p.add_argument("--inlet-base-resolution-factor", type=float, default=INLET_BASE_RESOLUTION_FACTOR, help=f"Base resolution factor for inlet (default: {INLET_BASE_RESOLUTION_FACTOR:.2f})")
    p.add_argument("--open-gui", default=False, action="store_true", help="Open the Gmsh GUI to visualize the mesh after generation")    
    args = p.parse_args(argv)
    
    # check if input file exists, has right extension and check/derive output path
    check_io_paths(args, ".brep", ".msh")
    
    # initialize reporter
    reporter = Reporter(args, __file__)
    reporter.start_log()

    # initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) # fix getBoundingBox to use STL bounds (more robust)
    gmsh.model.add("Leaf Plug Model")

    # suppress gmsh output if requested
    if args.suppress or args.suppress_gmsh:
        gmsh.option.setNumber("General.Terminal", 0)

    # import the BRep file
    tissue = kernel.importShapes(args.input_path) # usually [(3, 1)]
    kernel.synchronize()
    reporter.print(f"Imported tissue geometry from {args.input_path}")

    # Identify appropriate cylinder plug dimensions
    # shift to center at origin
    center, size = get_bbox(tissue)
    kernel.translate(tissue, -center[0], -center[1], -center[2])
    kernel.synchronize()
    # perform 2D meshing and extract the point furthest away from origin in xy-plane
    gmsh.model.mesh.generate(2)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)
    distances = np.linalg.norm(node_coords[:, :2], axis=1)
    max_distance = np.max(distances)
    reporter.print(f"Identified maximum radial distance in xy-plane: {max_distance:.3f}")

    # calculate cylinder geometry
    center, size   = get_bbox(tissue)
    bottom_z       = center[2] - size[2]*(0.5 + args.scm) # z-coordinate of the bottom cylinder surface
    height         = size[2]*(1 + args.scm + args.bm)
    # determine the appropriate dimensions for the cylinder plug
    bottom_surface = (center[0], center[1], bottom_z)
    axis           = (0, 0, height)
    radius         = (1+args.bm)*max_distance

    reporter.print("Calculated cylinder plug dimensions:")
    reporter.print(f"  bottom_surface: {bottom_surface}")
    reporter.print(f"  axis:           {axis}")
    reporter.print(f"  radius:         {radius:.3f}")

    # create the cylinder plug
    cylinder = [(3, kernel.addCylinder(
        *bottom_surface,
        *axis,
        radius))]
    kernel.synchronize()

    # perform boolean cut to create airspace
    airspace, _ = kernel.cut(cylinder, tissue, removeObject=True, removeTool=True)
    kernel.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    reporter.print(f"Created airspace by boolean cutting cylinder from tissue, resulting in # {len(volumes)} volume(s)")
    largest_volume = 0
    largest_volume_tag = None 
    for dim, tag in volumes:
        mass = kernel.getMass(dim, tag)
        if mass > largest_volume:
            largest_volume = mass
            largest_volume_tag = tag

    for dim, tag in volumes:
        if tag != largest_volume_tag:
            kernel.remove([(dim, tag)]) # recurvsive=True will remove all lower dimensional entities shared at the boundary

    kernel.synchronize()
    airspace = [(3, largest_volume_tag)]
    reporter.print(f"Retained largest volume as airspace: {airspace}")

    # Iteratively apply affine transformation to airspace to center bottom surface at origin and scale height to 1
    def transformation(center: tuple[float, float, float], size: tuple[float, float, float], target_size: float) -> list[float]:
        """ Generate affine transformation matrix to scale and translate entity """
        return [
            (target_size / size[2]), 0, 0, - center[0]           *(target_size / size[2]),
            0, (target_size / size[2]), 0, - center[1]           *(target_size / size[2]),
            0, 0, (target_size / size[2]), -(center[2]-size[2]/2)*(target_size / size[2]),
            0, 0, 0, 1
        ]

    def error(center: tuple[float, float, float], size: tuple[float, float, float], target_size: float) -> float:
        """ Calculate relative error in height """
        return (size[2] - target_size)/target_size

    iterations = iterative_affine_transformation(airspace, transformation, error, max_iterations=5, target_size=1.0)

    if not args.suppress: 
        center, size = get_bbox(airspace)
        reporter.print(f"Applied {iterations} affine transformation(s) to airspace. New center: {center}, New size: {size}")
        reporter.print("Finished transformations. Assigning physical groups...")
    #____________________________________________________
    
    # Assign Physical Groups

    # determine curved face tag 
    # OBS: this approach of identification by area only works if the curved area 2 pi r is unique up to tolerace
    # However, top and bottom surfaces will always be distinctly caught by the COM z-coordinate check below
    center, size = get_bbox(airspace)
    a = size[0]/2
    b = size[1]/2
    curved_area_target = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b))) # approximation of ellipse circumference to account for slight transform assymetry

    curved_area_found = []
    curved_area_tag = None
    top_area_tag = None 
    bottom_area_tag = None

    def iscurved(tag):
        area = kernel.getMass(2, tag)
        trigger = abs(area/curved_area_target - 1) <= args.tolerance
        if trigger: 
            curved_area_found.append(area)
            reporter.print(f"Curved area found, relative error: {area/curved_area_target - 1:.6f}")
        return trigger
    
    # airspace
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in airspace], 1, name="airspace")
    # surfaces
    surfaces = gmsh.model.getEntities(dim=2)
    reporter.print(f"{len(surfaces)} surfaces found")
    
    mesophyll_surface_tags = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0): 
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
            top_area_tag = tag
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
            bottom_area_tag = tag
        elif iscurved(tag):
            # curved surface of cylinder
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
        else:
            # other surfaces
            mesophyll_surface_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, mesophyll_surface_tags, 5, name="mesophyll_surfaces")
    
    assert len(curved_area_found) == 1, f"Error identifying curved face of cylinder. Found {len(curved_area_found)} curved faces with relative errors from target: {[area/curved_area_target - 1 for area in curved_area_found]}"
    
    reporter.print("Physical groups assigned. Proceeding to mesh generation...")
    
    assert top_area_tag is not None and bottom_area_tag is not None, "Error identifying top or bottom surface of cylinder"
    #____________________________________________________
    # Specify mesh size fields
    mesophyll_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(mesophyll_distance, "FacesList", mesophyll_surface_tags)
    mesophyll_threshold = gmsh.model.mesh.field.add("Threshold")    
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "IField", mesophyll_distance)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "LcMin", MINIMUM_RESOLUTION)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "LcMax", MAXIMUM_RESOLUTION)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "DistMin", MINIMUM_DISTANCE)
    gmsh.model.mesh.field.setNumber(mesophyll_threshold, "DistMax", MAXIMUM_DISTANCE)
    #
    inlet_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_distance, "FacesList", [bottom_area_tag, top_area_tag])
    inlet_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_threshold, "IField", inlet_distance)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMin", MINIMUM_RESOLUTION*INLET_BASE_RESOLUTION_FACTOR)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMax", MAXIMUM_RESOLUTION)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMin", MINIMUM_DISTANCE)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMax", MAXIMUM_DISTANCE)
    #
    minimum_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", [mesophyll_threshold, inlet_threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)
    kernel.synchronize()
    #____________________________________________________
    gmsh.model.mesh.generate(3)
    gmsh.write(args.output_path)

    reporter.print(f"Mesh written to {args.output_path}")
    
    if args.open_gui: 
        gmsh.fltk.run()

    gmsh.finalize()
    reporter.print("Gmsh finalized")
    reporter.end_log()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

