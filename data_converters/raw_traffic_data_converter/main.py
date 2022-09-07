from argparse import ArgumentParser
from pprint import pprint
import datetime
import os
import pickle 
import random
import time

from arg_processor import ArgProcessor
from data_converter import DataConverter


def main():
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' + 
                             'a string.')
    ini_path = parser.parse_args().config
    
    # Pass to ArgProcessor to read and process arguments 
    arg_processor = ArgProcessor(ini_path)
    
    # Handle different data processing modes 
    dc = DataConverter(arg_processor.mode,
                       arg_processor.in_path,
                       arg_processor.out_path)


if __name__ == "__main__":
    # Override print to add timestamps 
    old_print = print
    def timestamped_print(*args, **kwargs):
      old_print("%s%s%s\t" % ("[",str(datetime.datetime.now()),"]"), 
                *args, 
                **kwargs)
    print = timestamped_print

    
    start_dt = datetime.datetime.now()
    start_t = time.time() 
    print("START")
    main()
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("\n\n\n==================")
    print("END")
    print("Total time: " + str(end_t - start_t))
    
