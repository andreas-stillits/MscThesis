"""
filename.py

Description ...

Usage:
    python filename.py [options]

Options:
    --option <type>          Description of the option (default: value)

"""

import argparse
from mscthesis.utilities.reporter import Reporter
from mscthesis.utilities.checker import check_io_paths
# import statements ...


# default parameters
DEFAULT_OPTION = 42



def main(argv=None):
    p = argparse.ArgumentParser(description="Description ...")
    p.add_argument("input_path", type=str, help="Input path must be a .XYZ file")
    p.add_argument("--option", type=int, default=DEFAULT_OPTION, help=f"Description of the option (default: {DEFAULT_OPTION})")
    args = p.parse_args(argv)

    check_io_paths(args, ".xyz", ".abc") # input and output extensions
    reporter = Reporter(args, __file__)
    reporter.start_log()
    
    # main logic here
    #...

    reporter.end_log()
    
    return 0



if __name__ == "__main__":
    raise SystemExit(main())