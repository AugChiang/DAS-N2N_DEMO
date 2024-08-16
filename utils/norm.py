import numpy as np

def z_score_norm(arr):
    datamean = np.mean(arr)
    datastd = np.std(arr)
    # print(f"MEAN:{datamean}, STD: {datastd}")
    return (arr - datamean) / datastd

def min_max_scaler(arr):
    '''linear normalization to [0,1]'''
    m, M = np.min(arr), np.max(arr)
    R = M - m
    return (arr - m) / R

def min_max_scaler_symmetric(arr):
    '''linear normalization to [-1,1]'''
    return min_max_scaler(arr)*2 - 1