"""This module processes the arguments given by the .ini file"""

from decimal import Decimal
from distutils.util import strtobool
from pathlib import Path
import ast
import configparser
import os 
import shutil

class ArgProcessor():
    """Class that handles the .ini arguments""" 
    
    def __init__(self, ini_path):
        """
        Reads the arguments from the input .ini file and checks their validity
        
        Args:
            ini_path: The path to the input .ini file 
        """
        # Read the .ini file 
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        # Read the argument
        self.input_train_path   = Path(config['INPUT']['InputTrainPath'])
        self.input_val_path     = Path(config['INPUT']['InputValPath'])
        self.input_test_path    = Path(config['INPUT']['InputTestPath'])
        self.input_model_path   = Path(config['INPUT']['InputModelPath'])
        self.output_base_path   = Path(config['OUTPUT']['OutputBasePath'])
        
        self.model_name         = config['MODEL']['ModelName'].lower()
        self.head_size          = int(config['MODEL']['HeadSize'])
        self.embed_size         = int(config['MODEL']['EmbedSize'])
        self.num_heads          = int(config['MODEL']['NumHeads'])
        self.ff_dim             = int(config['MODEL']['FFDim'])
        self.num_trf_blocks     = int(config['MODEL']['NuMTrfBlocks'])
        self.mlp_units          = ast.literal_eval(config['MODEL']['MLPUnits'])
        self.dropout            = float(config['MODEL']['Dropout'])
        self.mlp_dropout        = float(config['MODEL']['MLPDropout'])
        
        self.gpu                = config['RESOURCE']['GPU']
        self.do_training        = strtobool(config['TRAINING']['DoTraining'])
        self.batch_size         = int(config['TRAINING']['BatchSize'])
        self.num_epochs         = int(config['TRAINING']['NumEpochs'])
        self.do_testing         = strtobool(config['TESTING']['DoTesting'].lower())
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(ini_path, self.output_base_path)
        