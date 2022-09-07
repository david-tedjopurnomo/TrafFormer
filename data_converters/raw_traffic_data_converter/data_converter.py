"""This module handles data conversion """

import abc
import h5py
import numpy as np
import os


class DataConverter():
    """
    Base class for the data converters
    """
    def __init__(self, mode, in_path, out_path):
        """
        Establish some class variables and read the dataset
        
        Args:
            mode (string): 
            in_path (pathlib Path):
            out_path (pathlib Path): 
        """
        in_file = h5py.File(in_path, "r")
        
        breakpoint()
    
        
        