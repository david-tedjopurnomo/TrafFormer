"""This module processes the arguments given by the .ini file"""

from pathlib import Path
from decimal import Decimal
import ast
import configparser
import os 

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
        
        # Read the arguments
        self.mode       = str(config['MODE']['Mode']).lower()
        self.in_path    = Path(config['INPUT']['InPath'])
        self.out_path   = Path(config['OUTPUT']['OutPath'])