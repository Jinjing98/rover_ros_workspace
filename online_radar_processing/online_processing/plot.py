import numpy as np
import traceback
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

def prepare_plot(params):
    """
    TODO: find a solution with a dsp_settings class
    Prepares the dimension, labels and scaling before actually injecting data.
    init: array for initial dimensions
    plot: kwarg for which map will be plotted

    returns
    image: image on the axes
    color_bar: color_bar with max value corresponding to the plot
    """
    plot = params("plots")
    fig, ax = plt.subplots()
    range_res = params("range_res")
    max_range = range_res * params("range")
    doppler_res = params("velocity_res")
    max_velocity = min_velocity = params("doppler") * doppler_res / 2
    min_velocity *= -1
    
    if plot == "rc":
        print("#################", range_res)
        extent = [0, max_range, 0, params("range")]
        print("####### extent rc:", extent)
    elif plot == "rd":
        print("####### range res and doppler res:", range_res, doppler_res)
        extent = [min_velocity, max_velocity, 0, max_range]
        init = np.ndarray((params("range"), params("doppler")))
        vmax = 100
        print("####### extent rd:", extent)
    elif plot == "md":
        ### axes extent is Bbox([[79.99999999999997, 52.8], [476.79999999999995, 422.4]])
        extent = [min_velocity, max_velocity, 0, params("micro_doppler")]
        init = np.ndarray((params("micro_doppler"), params("doppler")))
        vmax = 2000
        
    image = ax.imshow(init, origin="bottom", aspect="auto", vmin=0, vmax=vmax, extent=extent, interpolation=None)
    print(ax.dataLim)
    print(ax.viewLim)
    color_bar = plt.colorbar(image)
    return fig, image, color_bar


def plot_range_map(radar_map, image, frame_idx):
    image.set_data(range_map)    
    plt.title("Range vs. Chirps - Frame " + str(frame_idx))
    plt.pause(0.05)
    return image


def plot_range_doppler_map(doppler_map, image, frame_idx):
    """
    radar_cube: expected to have shape (range_bins, num_channels, doppler_bins)
    """
    #print("in plot: data type is", doppler_map.dtype)
    image.set_data(doppler_map)
    plt.title("Range-Doppler-Map - Frame " + str(frame_idx))
    plt.pause(0.05)
    return image


def plot_micro_doppler(micro_doppler, image, frame_idx):
    image.set_data(micro_doppler)
    plt.title("Micro-Doppler-Signature - Frame " + str(frame_idx))
    plt.pause(0.05)
    return image


def plotting(plot_pipe, params):
    """
    Prepares plots and recieves images to plot over pipe/queue
    params: radar configuration and parameters
    """
    frame_num = 0
    _, image, _ = prepare_plot(params)
    plots = params("plots")
    
    while(True):
        try:
            print("get images from pipe")
            images, index = plot_pipe.get(timeout=3)
        except:
            print(traceback.format_exc())
            print("Queue for plotting timed out!")
            break
        if plots=="rd":
            plot_range_doppler_map(images, image, index)
        elif plots=="md":
            plot_micro_doppler(images, image, index)
        else:
            plot_range_map(images, image, index)
        frame_num += 1

    
