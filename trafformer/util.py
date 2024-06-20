from sklearn.preprocessing import RobustScaler

import numpy as np

def robust_fit(data):
    """
    Fit robust scaler to data
    
    Args:
        data (numpy array): data to perform the scaler fitting on
        
    Returns:
        scaler (sklearn RobustScaler): fitted data scaler
    """
    data = np.expand_dims(data.flatten(), axis=1)
    scaler = RobustScaler().fit(data)
    return scaler 

    
def robust_transform(data, scaler):
    """
    Scale data using a provided scaler
    
    Args:
        data (numpy array): data to scale
        scaler (sklearn RobustScaler): scaler to be used
        
    Returns:
        data (numpy array): scaled data
    """
    data_ = np.expand_dims(data.flatten(), axis=1)
    return scaler.transform(data_).reshape(data.shape)

    
def robust_inverse_transform(data, scaler):
    """
    Perform inverse scaling of data using a provided scaler
    
    Args:
        data (numpy array): data to perform inverse scaler on 
        scaler (sklearn RobustScaler): scaler to be used
        
    Returns:
        data (numpy array): inverse-scaled data
    """
    data_ = np.expand_dims(data.flatten(), axis=1)
    return scaler.inverse_transform(data_).reshape(data.shape)