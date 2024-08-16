import matplotlib.pyplot as plt
import numpy as np

def plot_zoomed_fig(arr, zooming_region:list, saveAs="fig/pred.png"):
    x1, x2, y1, y2 = zooming_region
    vmin, vmax = -1, 1
    if y2 <= arr.shape[0] // 2:
        zoomed_fig_pos = [0.5, 0.5, 0.48, 0.48]
    else:
        zoomed_fig_pos = [0.5, 0.02, 0.48, 0.48]
    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(saveAs.split('/')[-1].split('.')[0].title())

    # Define the extent to fit your array
    extent = (0, arr.shape[1], 0, arr.shape[0])

    # Display the image
    ax.imshow(arr, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')

    # Create inset axis
    axins = ax.inset_axes(
        zoomed_fig_pos,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(arr, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)
    plt.savefig(saveAs)
    plt.show()

def plot_gaussian_fig(noised_test_input, testnormal,
                      pred1, g_pred1,
                      pred2, g_pred2,
                      saveAs="fig/gaussian.png"):
    vmin, vmax = -4, 4
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6), sharey=True, sharex=True)
    ax[0,0].set_title("Noisy Input")
    ax[0,1].set_title("DAS-N2N")
    ax[0,2].set_title("DAS-N2N (tuned)")
    ax[0,0].imshow(noised_test_input, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0,1].imshow(np.squeeze(pred1), cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0,2].imshow(np.squeeze(pred2), cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)

    ax[1,0].set_title("Gaussian")
    # ax[1,1].set_title("DAS-N2N")
    # ax[1,2].set_title("DAS-N2N (tuned)")
    ax[1,0].imshow(testnormal, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1,1].imshow(np.squeeze(g_pred1), cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    im = ax[1,2].imshow(np.squeeze(g_pred2), cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.savefig(saveAs)
    plt.show()

def plot_comparison(raw,
                    pred1,
                    pred2,
                    zooming_region:list,
                    saveAs:str="fig/comparison.png"):
    '''
    Plot the figure that compares predictions of two models (include raw data).
    Include zooming parts for comparison if given the region.

    Args:
        raw (_type_): input data (raw data).
        pred1 (_type_): prediction of model 1.
        pred2 (_type_): prediction of model 2.
        zooming_region (list): list of coordinates (x1, x2, y1, y2) for zooming.
        saveAs (str, optional): path to save figure. Defaults to "fig/comparison.png".
    '''
    assert len(zooming_region) == 4, "Only supports rectangle region."
    x1, x2, y1, y2 = zooming_region
    vmin, vmax = -1, 1
    if y2 <= raw.shape[0] // 2:
        zoomed_fig_pos = [0.5, 0.5, 0.48, 0.48]
    else:
        zoomed_fig_pos = [0.5, 0.02, 0.48, 0.48]
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14,4))
    # Define the extent to fit your array
    extent = (0, raw.shape[1], 0, raw.shape[0])

    # Display the image
    ax[0].set_title("Raw Data")
    ax[0].imshow(raw  , extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax[1].set_title("Pred")
    ax[1].imshow(pred1, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax[2].set_title("Pred (Tuned)")
    ax[2].imshow(pred2, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')

    # Create inset axis
    axins = ax[0].inset_axes(
        zoomed_fig_pos,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(raw, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax[0].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    axins = ax[1].inset_axes(
        zoomed_fig_pos,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(pred1, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax[1].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    axins = ax[2].inset_axes(
        zoomed_fig_pos,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(pred2, extent=extent, origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    ax[2].indicate_inset_zoom(axins, edgecolor="black", linewidth=2.)

    plt.savefig(saveAs)
    plt.show()