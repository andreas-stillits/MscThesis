# Meshing Pipeline

Utilities for turning a binary 3D voxel representation (.npy) of leaf-like mesophyll tissues into its boundary representation (.brep) such that it is compatible with CAD model kernels in tools like GMSH for downstream Finite Element Modeling.

Files:

- data: Contains raw .npy files with 3D numpy arrays and later derived .stl and .brep representations
- scripts:
    - npy_to_brep.py: 
        - automated pipeline from .npy to .brep. 
        - tunable via command line arguments (type $ python npy_to_brep.py -h, for help)
        - will produce an .stl and .brep file with the same root name as the passed .npy file
    - freecad_converter.py:
        - helper script to be executed by FreeCAD-daily in headless mode
        - manages conversion from surface meshes to solid boundary representations
- visual_inspection.ipynb:
    - temporary interface for quick visualization of .npy --> .stl --> .brep intermediate results
- voxel_example.ipynb
    - temporary interface for generating example voxel representations for development