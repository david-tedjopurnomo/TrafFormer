"""This module handles data processing """

from tensorflow.data import Dataset

from util import robust_fit, robust_transform, robust_inverse_transform

import abc
import numpy as np
import os
import pickle



class DataProcessor():
    """
    Data processor class that reads data from the input files and provides
    necessary preprocessing for the deep learning models. 
    """
    def __init__(self, train_path, val_path, test_path):
        """
        Read the dataset 
        
        Args:
            train_path (Path): path to the input training data 
            val_path (Path): path to the input validation data
            test_path (Path): path to the input test data
        """
        # Override print to add timestamps 
        
        train = np.load(train_path)
        val = np.load(val_path)
        test = np.load(test_path)
        
        print("Reading training data")
        self.train_x, self.train_y   = train['x'], train['y']
        print("Train x shape %s" % (str(self.train_x.shape)))
        print("Train y shape %s\n" % (str(self.train_y.shape)))
        
        print("Reading validation data")
        self.val_x,   self.val_y     = val['x'],   val['y']
        print("Validation x shape %s" % (str(self.val_x.shape)))
        print("Validation y shape %s\n" % (str(self.val_y.shape)))
        
        print("Reading test data")
        self.test_x,  self.test_y    = test['x'],  test['y']
        print("Test x shape %s" % (str(self.test_x.shape)))
        print("Test y shape %s\n" % (str(self.test_y.shape)))
        
        self.num_train = self.train_x.shape[0]
        self.num_val = self.val_x.shape[0]
        self.num_test = self.test_x.shape[0]
        
        
    def generate_data_tf(self, model_name, batch_size):
        """
        Generate training, validation and test datasets based on different  
        models. Different models require different preprocessing. Returns
        a tf.dataset 
        
        Args:
            model_name (String): name of the model
            
        Returns:
            data_dict (tf.Data): dictionary containing three items for the 
                                 training, validation and test datasets
        """
        print("LOADING DATA USING TF.DATASET") 
        if model_name == "trafformer_speed":
            # Keep only the traffic speed data
            train_x = self.train_x[:,:,:,[0,3]]
            train_y = self.train_y[:,:,:,0]
            val_x = self.val_x[:,:,:,[0,3]]
            val_y = self.val_y[:,:,:,0]
            test_x = self.test_x[:,:,:,[0,3]]
            test_y = self.test_y[:,:,:,0]
            train = Dataset.from_tensor_slices((train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True) 
            val = val.batch(batch_size)
            data_dict = {"train": train, "val": val, 
                         "test_x": test_x, "test_y":test_y}
        elif (model_name == "trafformer_full" or 
             model_name == "feedforwardnn" or 
             model_name == "stackedgru" or 
             model_name == "seq2seq"):
            train_y = self.train_y[:,:,:,0]
            val_y = self.val_y[:,:,:,0]
            test_y = self.test_y[:,:,:,0]
            train = Dataset.from_tensor_slices((self.train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((self.val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True)
            val = val.batch(batch_size)
            data_dict = {"train": train, "val": val, 
                         "test_x": self.test_x, "test_y":test_y}
        elif model_name == "trafformer_hour":
            train_y = self.train_y[:,:,:,0]
            val_y = self.val_y[:,:,:,0]
            test_y = self.test_y[:,:,:,0]
            train_x = self.train_x[:,:,:,0:3]
            val_x = self.val_x[:,:,:,0:3]
            test_x = self.test_x[:,:,:,0:3]
            train = Dataset.from_tensor_slices((train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True)
            val = val.batch(batch_size)
            data_dict = {"train": train, "val": val, 
                         "test_x": test_x, "test_y":test_y}
        elif model_name == "trafformer_day":
            train_y = self.train_y[:,:,:,0]
            val_y = self.val_y[:,:,:,0]
            test_y = self.test_y[:,:,:,0]
            train_x = self.train_x[:,:,:,3:]
            val_x = self.val_x[:,:,:,3:]
            test_x = self.test_x[:,:,:,3:]
            train = Dataset.from_tensor_slices((train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True)
            val = val.batch(batch_size)
            data_dict = {"train": train, "val": val, 
                         "test_x": test_x, "test_y":test_y}
        elif model_name == "trafformer_cyc":
            train_x = self.transform_cyclical(self.train_x)
            val_x = self.transform_cyclical(self.val_x)
            test_x = self.transform_cyclical(self.test_x)
            train_y = self.train_y[:,:,:,0]
            val_y = self.val_y[:,:,:,0]
            test_y = self.test_y[:,:,:,0]
            train = Dataset.from_tensor_slices((train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True)
            val = val.batch(batch_size)
            data_dict = {"train": train, "val": val, 
                         "test_x": test_x, "test_y":test_y}
        elif model_name == "trafformer_scale":
            train_speeds = self.train_x[:,:,:,[0,3]]
            val_speeds = self.val_x[:,:,:,[0,3]]
            train_scaler = robust_fit(train_speeds)
            val_scaler = robust_fit(val_speeds)
            train_x = self.train_x
            val_x = self.val_x
            train_x[:,:,:,0] = robust_transform(train_x[:,:,:,0], train_scaler)
            train_x[:,:,:,3] = robust_transform(train_x[:,:,:,3], train_scaler)
            val_x[:,:,:,0] = robust_transform(val_x[:,:,:,0], val_scaler)
            val_x[:,:,:,3] = robust_transform(val_x[:,:,:,3], val_scaler)
            
            train_y = robust_transform(self.train_y[:,:,:,0], train_scaler)
            val_y = robust_transform(self.val_y[:,:,:,0], val_scaler)
            test_y = self.test_y[:,:,:,0]
            
            train = Dataset.from_tensor_slices((train_x, train_y))
            train = train.shuffle(self.num_train, reshuffle_each_iteration=True)
            train = train.batch(batch_size)
            val = Dataset.from_tensor_slices((val_x, val_y))
            val = val.shuffle(self.num_val, reshuffle_each_iteration=True)
            val = val.batch(batch_size)
            print("SCALE")
            data_dict = {"train": train, "val": val, 
                         "test_x": self.test_x, "test_y":test_y}
        else:
            assert False, "Invalid model name %s" % (model_name)
        return data_dict
        
    def z_normalize(self):
        """
        Perform z-normalization
        
        Returns:
            mean (float): mean of the training dataset
            std (float): standard deviation of the training dataset
        """
        mean, std = np.mean(self.train_x), np.std(self.train_x)
        self.train_x = (self.train_x - mean) / std
        self.train_y = (self.train_y - mean) / std
        self.val_x = (self.val_x - mean) / std
        self.val_y = (self.val_y - mean) / std
        self.test_x = (self.test_x - mean) / std
        self.test_y = (self.test_y - mean) / std
        return mean, std
        
        
    def transform_cyclical(self, data):
        """
        Transform the day and hour data into cyclical features 
        
        Args:
            data (np array): data to be transformed
            
        Returns:
            data_cyc (np array): transformed data 
        """
        day_speed   = np.expand_dims(data[:,:,:,0], axis=3)
        day_minute  = np.expand_dims(data[:,:,:,1], axis=3)
        day_day     = np.expand_dims(data[:,:,:,2], axis=3)
        hour_speed  = np.expand_dims(data[:,:,:,3], axis=3)
        hour_minute = np.expand_dims(data[:,:,:,4], axis=3)
        hour_day    = np.expand_dims(data[:,:,:,5], axis=3)
        
        day_minute_sin = np.sin(2 * np.pi * day_minute / float(287))
        day_minute_cos = np.cos(2 * np.pi * day_minute / float(287))
        day_day_sin = np.sin(2 * np.pi * day_day / float(23))
        day_day_cos = np.cos(2 * np.pi * day_day / float(23))
        hour_minute_sin = np.sin(2 * np.pi * hour_minute / float(287))
        hour_minute_cos = np.cos(2 * np.pi * hour_minute / float(287))
        hour_day_sin = np.sin(2 * np.pi * hour_day / float(23))
        hour_day_cos = np.cos(2 * np.pi * hour_day / float(23))
        
        data_cyc = np.concatenate([day_speed, day_minute_sin, day_minute_cos,
                                   day_day_sin, day_day_cos,
                                   hour_speed, hour_minute_sin, hour_minute_cos,
                                   hour_day_sin, hour_day_cos], axis=3) 
        return data_cyc
        
        
        