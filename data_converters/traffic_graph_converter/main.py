from argparse import ArgumentParser
from pprint import pprint
import datetime
import os
import pickle 
import random
import time

from arg_processor import ArgProcessor
from weight_converter import WCVeritasYin, WCJianzhongQi


def main():
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' + 
                             'a string.')
    ini_path = parser.parse_args().config
    
    # Pass to ArgProcessor to read and process arguments 
    arg_processor = ArgProcessor(ini_path)
    
    # Read arguments
    mode = arg_processor.mode
    input_h5_path = arg_processor.input_h5_path
    input_weight_path = arg_processor.input_weight_path
    output_weight_path = arg_processor.output_weight_path
    
    # Process the weights 
    if mode == "veritasyin":
        wc = WCVeritasYin(input_h5_path,input_weight_path,output_weight_path)
    elif mode == "jianzhongqi":
        wc = WCJianzhongQi(input_h5_path,input_weight_path,output_weight_path)
    else:
        assert False, "Mode %s not implemented" % mode
    wc.convert_and_write_weights()

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
    
