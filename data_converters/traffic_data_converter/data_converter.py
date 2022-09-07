"""This module handles data conversion """

import abc
import numpy as np
import os
import pickle


class DataConverter():
    """
    Base class for the converters. The converters perform data conversion from 
    the DCRNN train, val, test to a format compatible with various baselines
    """
    def __init__(self, train_path, val_path, test_path, out_path):
        """
        Establish some class variables and read the dataset
        
        Args:
            train_path (Path): path to the input training data 
            val_path (Path): path to the input validation data
            test_path (Path): path to the input test data
            out_path (Path): path to the output directory
        """
        self.train = np.load(train_path)
        self.val = np.load(val_path)
        self.test = np.load(test_path)
        
        self.out_path = out_path
    
    
    @abc.abstractmethod 
    def convert_and_write_data(self):
        """
        Convert the DCRNN data to a format compatible with various baseline
        models used.
        """
        return 

    
    def write_files(self, train, val, test, out_path):
        """
        Write the train, val and test set files to the output directory. 
        Assumes that these files are numpy arrays.
        
        Args:
            train (numpy array): numpy array containing training data
            val (numpy array): numpy array containing validation data 
            test (numpy array): numpy array containing test data 
            out_path (pathlib Path): path to the output directory
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(out_path / "train", train)
        np.save(out_path / "val", val)
        np.save(out_path / "test", test)


class DCVeritasYin(DataConverter):
    """
    Convert the DCRNN data to the "veritasyin" format
    
    Paper: https://arxiv.org/pdf/1709.04875v4.pdf
    Code: https://github.com/VeritasYin/STGCN_IJCAI-18
    """
    def convert_and_write_data(self):
        train_x, train_y = self.train['x'], self.train['y']
        val_x, val_y = self.val['x'], self.val['y']
        test_x, test_y = self.test['x'], self.test['y']
        
        # The DCRNN data has 2 values for the 4-th dimension for the ttaffic 
        # flow reading and the scaled timestamp, respectively. We only need 
        # the former here
        train_x = self.__trunc_hr(train_x)
        train_y = self.__trunc_hr(train_y)
        val_x, val_y = self.__trunc_hr(val_x), self.__trunc_hr(val_y)
        test_x, test_y = self.__trunc_hr(test_x), self.__trunc_hr(test_y)
        
        # Merge the X and Y data on the sequence (2nd) dimension
        train_xy = np.concatenate((train_x, train_y), axis=1)
        val_xy = np.concatenate((val_x, val_y), axis=1)
        test_xy = np.concatenate((test_x, test_y), axis=1)
        
        # Z normalize the data
        train_xy, train_mean, train_std = self.__z_normalize(train_xy)
        val_xy, val_mean, val_std = self.__z_normalize(val_xy)
        test_xy, test_mean, test_std = self.__z_normalize(test_xy)
        
        # Write the files to output file
        stats = {'train_mean' : train_mean, 'train_std': train_std,
                 'val_mean'   : val_mean  , 'val_std'  : val_std,
                 'test_mean'  : test_mean , 'test_std' : test_std}
        self.write_files(train_xy, val_xy, test_xy, stats, self.out_path)
        
        
    def write_files(self, train, val, test, stats, out_path):
        """
        Write the train, val and test set files to the output directory. 
        Assumes that these files are numpy arrays.
        
        Args:
            train (numpy array): numpy array containing training data
            val (numpy array): numpy array containing validation data 
            test (numpy array): numpy array containing test data 
            stats (dict): dict containing the mean and std of the dataset
            out_path (pathlib Path): path to the output directory
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(out_path / "stats.pkl", "wb") as f:
            pickle.dump(stats, f)
        np.save(out_path / "train", train)
        np.save(out_path / "val", val)
        np.save(out_path / "test", test) 
        
        
    def __z_normalize(self, data):
        """
        Z-normalize the dataset
        
        Args:
            data (np.array): input DCRNN data
            
        Returns:
            data (np.array): Z-normalized input DCRNN data
            mean (float): mean of the data
            std (float): standard deviation of the data
        """
        mean, std = np.mean(data), np.std(data)
        data = (data - mean) / std
        return data, mean, std
        
        
    def __trunc_hr(self, data):
        """
        Remove the hour data from an input numpy array of DCRNN data
        
        Args:
            data (np.array): input DCRNN data
            
        Return:
            data (np.array): DCRNN data with the hour information removed
        """
        [num, seq, det, _ ] = data.shape
        data = data[:,:,:,0]
        data = data.reshape((num, seq, det, 1))
        return data
        
        
        