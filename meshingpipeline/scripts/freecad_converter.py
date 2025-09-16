import os, sys
import Mesh, Part 

def stl_to_brep(stl_path, brep_path, tolerance=0.05):
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"Input STL not found: {stl_path}")
    mesh = Mesh.Mesh(stl_path)
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, tolerance)
    try:
        # Try to create a solid directly
        solid = Part.makeSolid(shape)
    except Exception:
        try:
            # if that fails, try to create a solid from the shell
            shell = Part.Shell(shape.Faces)
            solid = Part.Solid(shell)
        except Exception:
            shape.exportBrep(brep_path)
            print("Could not form solid. Exporting compound/shape to", brep_path)
            return
    # export solid to brep if successful
    solid.exportBrep(brep_path)
    print('Exported BRep:', brep_path)


def main():
    # expects environment variables INPUT_STL and OUTPUT_BREP for file names
    stl = os.environ.get('INPUT_STL')
    brep = os.environ.get('OUTPUT_BREP')
    # if not declared, exit without export attempt
    if not stl or not brep:
        print('Please set INPUT_STL and OUTPUT_BREP environment variables!', file=sys.stderr)
        return
    stl_to_brep(stl, brep, tolerance=0.05)


main()