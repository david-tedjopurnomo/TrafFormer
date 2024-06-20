from argparse import ArgumentParser
from pprint import pprint
import datetime
import os
import pickle 
import sys
import random
import time

from arg_processor import ArgProcessor
from data_processor import DataProcessor
from dnn_models import TrafFormerSpeed, TrafFormerFull, TrafFormerCyc
from dnn_models import FeedforwardNN, StackedGRU, Seq2Seq, TrafFormerSingle
from dnn_models_fc import TrafFormer
from model_operator import train_model, test_model 


def main():
    # Process arguments from ini file
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' +  
                             'a string.')
    ini_path = parser.parse_args().config
    arg_processor = ArgProcessor(ini_path)
    
    # Set the gpu to be used 
    os.environ["CUDA_VISIBLE_DEVICES"] = arg_processor.gpu
    
    # Process and generate data 
    dp = DataProcessor(arg_processor.input_train_path,
                       arg_processor.input_val_path,
                       arg_processor.input_test_path)
    #mean, std = dp.z_normalize()
    model_name = arg_processor.model_name
    data_dict = dp.generate_data_tf(model_name, arg_processor.batch_size)
    shape = data_dict['test_x'].shape[1:]

    # Load model parameters
    model_name      = arg_processor.model_name
    head_size       = arg_processor.head_size
    embed_size      = arg_processor.embed_size
    num_heads       = arg_processor.num_heads
    ff_dim          = arg_processor.ff_dim
    num_trf_blocks  = arg_processor.num_trf_blocks
    mlp_units       = arg_processor.mlp_units
    dropout         = arg_processor.dropout
    mlp_dropout     = arg_processor.mlp_dropout
    
    # Create the model 
    if arg_processor.model_name == "trafformer_speed":
        traff = TrafFormerSpeed()
        model = traff.build_model(in_out_shape = shape, 
                                head_size = head_size, 
                                num_heads = num_heads, 
                                ff_dim = ff_dim, 
                                num_transformer_blocks = num_trf_blocks, 
                                mlp_units = mlp_units,
                                dropout = dropout,
                                mlp_dropout = mlp_dropout)
    elif (arg_processor.model_name == "trafformer_full" or 
          arg_processor.model_name == "trafformer_scale"):
        traff = TrafFormerFull()
        model = traff.build_model(in_out_shape = shape, 
                                head_size = head_size, 
                                embed_size = embed_size,
                                num_heads = num_heads, 
                                ff_dim = ff_dim, 
                                num_transformer_blocks = num_trf_blocks, 
                                mlp_units = mlp_units,
                                dropout = dropout,
                                mlp_dropout = mlp_dropout)
    elif (arg_processor.model_name == "trafformer_hour" or 
          arg_processor.model_name =="trafformer_day"):
        traff = TrafFormerSingle()
        model = traff.build_model(in_out_shape = shape, 
                                head_size = head_size, 
                                embed_size = embed_size,
                                num_heads = num_heads, 
                                ff_dim = ff_dim, 
                                num_transformer_blocks = num_trf_blocks, 
                                mlp_units = mlp_units,
                                dropout = dropout,
                                mlp_dropout = mlp_dropout)
    elif arg_processor.model_name == "trafformer_cyc":
        traff = TrafFormerCyc()
        model = traff.build_model(in_out_shape = shape, 
                                head_size = head_size, 
                                num_heads = num_heads, 
                                ff_dim = ff_dim, 
                                num_transformer_blocks = num_trf_blocks, 
                                mlp_units = mlp_units,
                                dropout = dropout,
                                mlp_dropout = mlp_dropout)
    elif arg_processor.model_name == "feedforwardnn":
        traff = FeedforwardNN()
        model = traff.build_model(shape, embed_size, mlp_units)
    elif arg_processor.model_name == "stackedgru":
        traff = StackedGRU()
        model = traff.build_model(shape, embed_size, mlp_units)
    elif arg_processor.model_name == "seq2seq":
        traff = Seq2Seq()
        model = traff.build_model(shape, embed_size, mlp_units)
    else:
        assert False, "Invalid model name %s" % arg_processor.model_name
   
    # Train and test, if needed
    out_path = arg_processor.output_base_path
    if arg_processor.do_training:
        train_model(model, data_dict, out_path,
                    arg_processor.batch_size,
                    arg_processor.num_epochs)
    if arg_processor.do_testing: 
        results_dict = test_model(model, arg_processor.input_model_path, 
                                  data_dict, arg_processor.batch_size, 
                                  model_name)
    
    # Write results
    print(results_dict)
    with open(arg_processor.output_base_path / "results.txt", "w") as f:
        f.write(str(results_dict))
         
        

if __name__ == "__main__":
    start_dt = datetime.datetime.now()
    start_t = time.time() 
    print("START")
    main()
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("\n\n\n==================")
    print("END")
    print("Total time: " + str(end_t - start_t))
    
