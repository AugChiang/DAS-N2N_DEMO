import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
from glob import glob
import numpy as np
from utils.norm import z_score_norm
from utils.plot import plot_comparison, plot_gaussian_fig, plot_zoomed_fig
from utils.rearrange import arr_to_2D

# print(tf.__version__)

# Define zooming region
# x1, x2, y1, y2
zoom_region = [300, 556, 500, 1012]

def gen_gaussian_test_input():
    # gen random test samples to show the basic denoising ability
    testnormal = np.random.normal(scale=2, size=(128,96))
    testinput = np.zeros((128,96))
    testinput[:,20:30] = 1
    testinput[:,70:90] = -1
    return testinput, testnormal

def main():
    # weight path
    model_path = r"weights/dasn2n_model"
    tuned_model_path = r"weights/TunedModel.h5"

    # DAS-N2N accepts input shape of (128,96)
    file_paths = glob(os.path.join('test_data/*.npy'))
    test_data = np.load(file_paths[0])
    print("Loading test data ... ")

    nr, nc, h, w = test_data.shape
    test_data = np.reshape(test_data, (-1, h, w))
    test_data = z_score_norm(test_data)
   
    model = tf.keras.models.load_model(model_path)
    tuned_model = tf.keras.models.load_model(tuned_model_path)

    # plot_model(model, to_file='fig/model_arch.png', rankdir='LR')

    # gen random test samples to show the basic denoising ability
    testinput, testnormal = gen_gaussian_test_input()
    noised_test_input = testinput + testnormal

    # prediction of Gaussian noises
    pred = tuned_model.predict(np.expand_dims(noised_test_input,0))
    g_pred = tuned_model.predict(np.expand_dims(testnormal,0))
    tuned_pred = tuned_model.predict(np.expand_dims(noised_test_input,0))
    tuned_g_pred = tuned_model.predict(np.expand_dims(testnormal,0))
    plot_gaussian_fig(noised_test_input, testnormal,
                      pred, g_pred,
                      tuned_pred, tuned_g_pred)

    # prediction of DAS data
    plot_raw = arr_to_2D(test_data, nr=nr, nc=nc)[::10,:]  # downsample for plotting
    # plot_zoomed_fig(plot_raw, zoom_region, "fig/raw.png")

    pred = model.predict(test_data)
    plot_arr = arr_to_2D(pred, nr=nr, nc=nc)[::10,:] # downsample for plotting
    # plot_zoomed_fig(plot_arr, zoom_region, "fig/pred_no_tuned.png")

    tuned_pred = tuned_model.predict(test_data)
    plot_tuned = arr_to_2D(tuned_pred, nr=nr, nc=nc)[::10,:] # downsample for plotting ~(3000,1000)
    # plot_zoomed_fig(plot_tuned, zoom_region, "fig/pred_tuned.png")

    plot_comparison(plot_raw, plot_arr, plot_tuned, zoom_region)

if __name__ == '__main__':
    main()