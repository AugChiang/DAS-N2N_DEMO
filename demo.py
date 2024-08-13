import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# Define zoom region
# x1, x2, y1, y2
region = [300, 556, 500, 1012]


def load_test_data(file_path:str):
    '''load the .npy test data'''
    filename = file_path.split('\\')[-1]
    arr = np.load(file_path)
    H, W = arr.shape[-2], arr.shape[-1]
    arr = np.reshape(arr, (-1, H, W))
    return filename, arr
        
def normalize(arr):
    arr = np.squeeze(arr)
    datamean = np.mean(arr)
    datastd = np.std(arr)
    # print(f"MEAN:{datamean}, STD: {datastd}")
    return (arr - datamean) / datastd

def arr_recover(arr):
    arr = np.reshape(arr, (235,-1,128,96))
    nr, nc, h, w = arr.shape
    arr = arr.swapaxes(1,2)
    arr = np.reshape(arr, (nr*h, nc*w))
    return arr

def plot_zoomed_fig(arr, x1, x2, y1, y2, saveAs="fig/pred.png"):
    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(saveAs.split('/')[-1].split('.')[0].title())

    # Define the extent to fit your array
    extent = (0, arr.shape[1], 0, arr.shape[0])

    # Display the image
    ax.imshow(arr, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')

    # Create inset axis
    axins = ax.inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(arr, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)
    plt.savefig(saveAs)
    plt.show()

def gen_gaussian_test_input():
    # gen random test samples to show the basic denoising ability
    testnormal = np.random.normal(scale=2, size=(128,96))
    testinput = np.zeros((128,96))
    testinput[:,20:30] = 1
    testinput[:,70:90] = -1
    return testinput, testnormal

def plot_gaussian_test_input(noised_test_input, testnormal, pred, g_pred, saveAs="fig/gaussian.png"):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    ax[0,0].set_title("Noisy Input")
    ax[0,1].set_title("DAS-N2N Pred")
    ax[0,0].imshow(noised_test_input, cmap='seismic', aspect='auto', vmin=-4, vmax=4)
    ax[0,1].imshow(np.squeeze(pred), cmap='seismic', aspect='auto', vmin=-4, vmax=4)

    ax[1,0].set_title("Gaussian")
    ax[1,1].set_title("DAS-N2N Pred")
    ax[1,0].imshow(testnormal, cmap='seismic', aspect='auto', vmin=-4, vmax=4)
    im = ax[1,1].imshow(np.squeeze(g_pred), cmap='seismic', aspect='auto', vmin=-4, vmax=4)

    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.savefig(saveAs)
    plt.show()

def plot_compare(x1,x2,y1,y2, raw, pred1, pred2, saveAs="fig/comparison.png"):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14,4))
    # Define the extent to fit your array
    extent = (0, raw.shape[1], 0, raw.shape[0])

    # Display the image
    ax[0].set_title("Raw Data")
    ax[0].imshow(raw  , extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax[1].set_title("Pred")
    ax[1].imshow(pred1, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax[2].set_title("Pred (Tuned)")
    ax[2].imshow(pred2, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')

    # Create inset axis
    axins = ax[0].inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(raw, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax[0].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    axins = ax[1].inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(pred1, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax[1].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    axins = ax[2].inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(pred2, extent=extent, origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax[2].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    plt.savefig(saveAs)
    plt.show()

def main():
    # weight path
    model_path = r"weights/dasn2n_model"
    tuned_model_path = r"weights/TunedModel.h5"

    # DAS-N2N accepts input shape of (128,96)
    file_paths = glob(os.path.join('test_data/*.npy'))
    dataname, test_data = load_test_data(file_paths[0])
    print("Loading ... ", dataname)
    test_data = normalize(test_data)
   
    model = tf.keras.models.load_model(model_path)
    tuned_model = tf.keras.models.load_model(tuned_model_path)

    # plot_model(model, to_file='fig/model_arch.png', rankdir='LR')

    # gen random test samples to show the basic denoising ability
    testinput, testnormal = gen_gaussian_test_input()
    noised_test_input = testinput + testnormal

    # predict (denoise)
    pred = model.predict(np.expand_dims(noised_test_input,0))
    g_pred = model.predict(np.expand_dims(testnormal,0))
    plot_gaussian_test_input(noised_test_input, testnormal, pred, g_pred)

    plot_raw = arr_recover(test_data)[::10,:]  # downsample for plotting
    # plot_zoomed_fig(plot_raw, *region, "fig/raw.png")

    pred = model.predict(test_data)
    plot_arr = arr_recover(pred)[::10,:] # downsample for plotting
    # plot_zoomed_fig(plot_arr, *region, "fig/pred_no_tuned.png")

    tuned_pred = tuned_model.predict(test_data)
    plot_tuned = arr_recover(tuned_pred)[::10,:] # downsample for plotting ~(3000,1000)
    # plot_zoomed_fig(plot_tuned, *region, "fig/pred_tuned.png")

    plot_compare(*region, plot_raw, plot_arr, plot_tuned)

if __name__ == '__main__':
    main()