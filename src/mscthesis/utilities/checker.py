"""  
checker.py

Standard snippet to check:
- if a file exists
- if a file has the correct extension
- derive output path if not provided

Usage:
    from mscthesis.utilities.checker import check_io_paths

    check_io_paths(args, ".abc", ".xyz")

"""

import os
import argparse 

def check_io_paths(args: argparse.Namespace, input_extension: str, output_extension: str) -> None:
    # Check if input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file {args.input_path} does not exist.")

    # Check if input file has the correct extension
    if not args.input_path.lower().endswith(input_extension):
        raise ValueError(f"Input file {args.input_path} is not a {input_extension} file.")

    # Derive output path if not provided
    if args.output_path is None:
        args.output_path = os.path.splitext(args.input_path)[0] + output_extension
    elif not args.output_path.lower().endswith(output_extension):
        raise ValueError(f"Output file {args.output_path} is not a {output_extension} file.")
