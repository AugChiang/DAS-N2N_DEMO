import numpy as np

def arr_to_2D(arr, nr, nc):
    assert len(arr.shape) == 3 # num_row*num_col, tile_height, tile_width
    h, w = arr.shape[1], arr.shape[2]
    arr = np.reshape(arr, (nr, nc, h, w))
    arr = arr.swapaxes(1,2)
    arr = np.reshape(arr, (nr*h, nc*w))
    return arr