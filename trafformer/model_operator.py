"""
This module contains model operation functions such as training and evaluation
"""


from tensorflow.keras import backend as K
from tensorflow.keras import callbacks 

import numpy as np
import tensorflow as tf

def train_model(model, data, out_path, batch_size, num_epochs):
    """
    Train a model
    
    Args:
        model (keras model): model to be trained
        data (dict): dictionary containing the datasets
        out_path (pathlib Path): path to the output directory
        batch_size (int): batch size
        num_epochs (int): number of epochs
    """
    checkpoint_path = out_path / "checkpoint.h5"
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_freq="epoch",
        save_best_only=True)
        
    logger_callback = callbacks.CSVLogger(out_path / 'training.log')
    early_stop_callback = callbacks.EarlyStopping(patience=2) 
    scheduler_callback = callbacks.LearningRateScheduler(exp_schedule)
    
    if 'train_x' in data:
        train_x, train_y = data['train_x'], data['train_y']
        val_x, val_y = data['val_x'], data['val_y'] 
        model.compile(optimizer="Adam", loss=masked_mse, metrics=["mae"])
        history = model.fit(x=train_x, y=train_y, batch_size=batch_size, 
                            epochs=num_epochs,
                            validation_data=(val_x, val_y), 
                            callbacks=[checkpoint_callback, 
                                       logger_callback,
                                       early_stop_callback,
                                       scheduler_callback])
    else:
        train, val = data['train'], data['val']
        model.compile(optimizer="Adam", loss=masked_mse, metrics=["mae"])
        history = model.fit(train, epochs=num_epochs, validation_data=val, 
                            callbacks=[checkpoint_callback, 
                                       logger_callback,
                                       early_stop_callback,
                                       scheduler_callback])
        
                            
                        
def test_model(model, weights_path, data, batch_size):
    """
    Test a model
    
    Args:
        model (keras model): model to be tested
        weights_path (pathlib Path): path to the model weights to load
        data (dict): dictionary containing the datasets
        batch_size (int): batch size
        
    Results:
        result_dict (dict): dictionary containing the results
    """
    model.load_weights(weights_path)
    test_x, test_y = data['test_x'], data['test_y']
    pred_y = model.predict(test_x, batch_size=batch_size, verbose=1)
    results_dict = {}
    index_hour = [(2,"6 hours"),(5,"12 hours"),(8,"18 hours"),(11,"24 hours")]
    for i, hour in index_hour:
        rmse, mae, mape = cal_error(test_y, pred_y, i)
        results_dict[hour] = [rmse, mae, mape]
    return results_dict
    
    
def cal_error(g_t, prediction, p):
    """
    Calculate error between the prediction and the ground truth.
    
    Args:
        g_t (np array): ground trth
        prediction (np array): prediction
        p (integer): index for the prediction
        
    Returns:
        rmse (float): root mean squared error
        mae (float): mean absolute error
        mape (float): mean absolute percentage error 
    """
    g_t = g_t[:,p,:]
    prediction = prediction[:,p,:,0]
    
    # Only choose nonzero values, in line with the other baselines
    nz_index = np.nonzero(g_t)
    g_t = g_t[nz_index]
    prediction = prediction[nz_index]
    g_t = np.float32(g_t)
    prediction = np.float32(prediction)
    
    abe = np.fabs(g_t - prediction)
    rmse = np.sqrt(np.mean(abe * abe))
    mae = np.mean(abe)
    mape = np.mean(abe / g_t)
    return rmse, mae, mape
    
    
def exp_schedule(epoch, lr):
    """
    Exponential learning rate schedule
    
    Args:
        epoch (int): current epoch
        lr (float): learning rate
        
    Returns:
        lr (float): updated learning rate 
    """
    if epoch < 10:
        return lr
    else:
        lr = lr * tf.math.exp(-0.1)
        return lr 
        
        
def masked_mse(y_true, y_pred):
    """
    An MSE loss that ignores zero values from the ground truth label.
    """
    return K.sum(K.square(y_pred*K.cast(y_true>0, "float32") - y_true), axis=-1) / K.sum(K.cast(y_true>0, "float32")) 
    
    
    
    
    