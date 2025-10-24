""" 
reporter.py

Class for managing printing and logging within the meshing pipeline.
The Reporter will log date, time, I/O, executing script path, all variables,
and printed messages to a log file, and optionally
suppress console output if desired.

Usage:
    from mscthesis.utilities.reporter import Reporter
    reporter = Reporter(args, __file__)
    reporter.start_log()
    ________________

    main script ...
    
    reporter.print("...") as you would usually print
    
    ________________
    reporter.end_log()

OBS: args must have attributes 'input_path', and 'output_path'.
    
"""

import os 
import argparse
from datetime import datetime
import time

class Reporter:
    def __init__(self, args: argparse.Namespace, parent: str) -> None:
        self.log_path = args.output_path if hasattr(args, 'output_path') else "default_reporter.txt"
        self.log_path = os.path.splitext(self.log_path)[0] # remove extension if any
        self.log_path += "_" + os.path.splitext(os.path.basename(parent))[0] + "_log.txt"
        self.args = args
        self.parent = parent
        self.start_time = None 
        self.end_time = None
        self.suppress_console_output = args.suppress if hasattr(args, 'suppress') else False
        if self.log_path:
            dir_name = os.path.dirname(self.log_path) # get full directory path
            if dir_name: # make it only if dir_name is not empty
                os.makedirs(dir_name, exist_ok=True)
            self.log_file = open(self.log_path, 'w')  # overwrite if exists, create if not
        else:
            self.log_file = None
        
    def start_log(self) -> None:
        if self.log_file and self.args:
            # record start time
            self.start_time = time.perf_counter()
            # write the date and time to the log file
            self.log_file.write(f"Log created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            # write the input path to the log file - insert some tabulation for readability
            self.log_file.write(f"Input path:      {os.path.abspath(self.args.input_path) if hasattr(self.args, 'input_path') else 'N/A'}\n")
            # write the path of the file that called this reporter
            self.log_file.write(f"Executed script: {os.path.abspath(self.parent)}\n")
            # write the output path to the log file
            self.log_file.write(f"Output path:     {os.path.abspath(self.args.output_path) if hasattr(self.args, 'output_path') else 'N/A'}\n")
            # write all arguments to the log file except input_path and output_path
            self.log_file.write("Arguments:\n")
            for arg, value in vars(self.args).items():
                if arg not in ["input_path", "output_path"]:
                    self.log_file.write(f"  {arg}: {value}\n")
            self.log_file.write("\n") # add an extra newline for better readability
            self.log_file.write("----- Begin Log -----\n\n")
            self.log_file.flush() # forces to main memory immediately
        else:
            raise ValueError("Cannot log arguments: log_file or args is None")


    def print(self, message: str) -> None:
        if not self.suppress_console_output:
            print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush() # forces to main memory immediately

    def end_log(self) -> None:
        if self.log_file:
            # record end time
            self.end_time = time.perf_counter()
            elapsed_time = self.end_time - self.start_time if self.start_time else 0.0
            msg = f"\nTotal elapsed time: {elapsed_time:.3f} seconds\n"
            print(msg)
            self.log_file.write(msg)
            self.log_file.write("\n----- End of Log -----\n")
            self.log_file.flush()
            self.log_file.close()
