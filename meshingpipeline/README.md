# Meshing Pipeline

Utilities for turning a binary 3D voxel representation (.npy) of leaf-like mesophyll tissues into its boundary representation (.brep) such that it is compatible with CAD model kernels in tools like GMSH for downstream Finite Element Modeling.

Files:

    - npy_to_brep.py: 
        - automated pipeline from .npy to .brep. 
        - tunable via command line arguments (type $ python npy_to_brep.py -h, for help)
        - will produce an .stl and .brep file with the same root name as the passed .npy file
    - freecad_converter.py:
        - helper script to be executed by FreeCAD-daily in headless mode
        - manages conversion from surface meshes to solid boundary representations
    - temporary interface for quick visualization of .npy --> .stl --> .brep intermediate results
    - temporary interface for generating example voxel representations for development


## Structure

- **data/**  
    Contains sample files for mesh processing:
    - `n_spheres.brep` — BREP format geometry.
    - `n_spheres.npy` — Numpy array data.
    - `n_spheres.stl` — STL mesh file.

- **scripts/**  
    Python scripts for mesh conversion and processing:
    - `freecad_converter.py` — Converts mesh files using FreeCAD.
    - `npy_to_brep.py` — Converts Numpy arrays to BREP format.

- **Notebooks**
    - `visual_inspection.ipynb` — Jupyter notebook for visualizing and inspecting mesh data.
    - `voxel_example.ipynb` — Example notebook for voxel-based mesh operations.

## Usage

1. Place your .npy files in the `data/` directory.
2. Use the scripts in `scripts/` to convert or process your files as needed.
3. Open the notebooks to visualize results or experiment with mesh operations.

## Requirements

- Python 3.12+
- FreeCAD (for conversion scripts)
- Open3D
- Numpy

Install dependencies with:
```bash
pip install numpy open3d
```

## Notes

- The `__pycache__/` directories contain compiled Python files.
- For FreeCAD-based operations, ensure FreeCAD is installed and accessible in your environment.