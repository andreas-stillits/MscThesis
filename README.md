src/ layout project

root/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ src/
│  └─ cfix/                         # your importable package
│     ├─ __init__.py
│     ├─ common/
│     │  ├─ __init__.py
│     │  └─ io.py
│     ├─ pipeline_a/
│     │  ├─ __init__.py
│     │  ├─ modules/
│     │  │  ├─ __init__.py
│     │  │  └─ mesh_ops.py
│     │  └─ data/                  # small packaged assets (optional)
│     │     └─ seed.msh
│     └─ pipeline_b/
│        ├─ __init__.py
│        ├─ modules/
│        │  ├─ __init__.py
│        │  └─ postproc.py
│        └─ data/
├─ tests/
│  ├─ conftest.py
│  └─ test_smoke.py
├─ scripts/                         # runnable CLI entry scripts (optional)
│  └─ run_pipeline_a.py
├─ .ruff.toml                       # linting (optional but recommended)
├─ mypy.ini                         # static typing (optional)
└─ pytest.ini                       # test runner config (optional)

Installation:

Create a virtual environment with: 
    $ conda env create -f environment.yml

This installs dependencies and the package contained in this repository
The "pip install -e ." is responsible of the package install when run in the directory containing our .toml file. The -e flag makes sure that we dont need to re-install every time we update the source code - they are linked instead of copied. Another installation is only needed if we create a new virtual environment.
