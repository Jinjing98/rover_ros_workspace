import matplotlib.pyplot as plt
import numpy as np
import time
import dsp
import utils
from mmwave.dsp import cfar
import mmwave.dsp.utils as ut
from online_capture import Parameters
import matplotlib.pyplot as plt


def plot_test():
    par = Parameters()
    _, im, _ = plot.prepare_plot(par, "md")
    data = np.zeros((128, 64))
    data[0, :]  = 2000
    data[50, :] = 2000
    #data[, :] = 2000
    data[-1, :] = 2000
    print(data)
    im.set_data(data)
    plt.pause(0.05)
    time.sleep(10)


def cfar_test(data):
    """
    det_obj = cfar.ca(data, l_bound=0)
    print(det_obj)
    return det_obj
    """
    det_obj2D = np.ndarray(data.shape, dtype=bool) # 128 x 64
    idx = 0
    #det_obj2D = cfar.ca(data, l_bound=900)
    
    for doppler in data: # doppler is actually a vector of 64 elements // 128 iterations in the loop
        det_obj = cfar.ca(doppler, guard_len=2, noise_len=10, mode='constant', l_bound=900)
        det_obj2D[idx, :] = det_obj
        idx += 1
    
    return det_obj2D

def cfar_threshold_test(data):
    thresholds = np.ndarray(data.shape, dtype=np.float)
    idx = 0
    for doppler in data:
        threshold, noise_floor = cfar.ca_(doppler, l_bound=1000)
        thresholds[idx, :] = threshold
        idx += 1
    return thresholds    

params = Parameters()
data = np.load("raw.npz")["arr_0"][:70,:,:,:]
print("shape of loaded data is", data.shape)
num_frames = data.shape[0]

frame_idx = 0

FRAME_TIME = 128 * params("chirp_time")
VEL_RESOLUTION = params("velocity_res")

scaled_range = utils.window_scalar(params.get("samples"), utils.Window.HANNING)
scaled_doppler = utils.window_scalar(params.get("doppler"), utils.Window.HANNING)
#plt.imshow(np.random.rand(16,16) > 0.5)
#plt.pause(5)
fig, axes = plt.subplots(2)

for frame in data:
    #print("shape of single frame is", frame.shape)
    radar_cube = dsp.separate_tx(frame, 2)
    range_cube = dsp.rc_fft(radar_cube, utils.Window.HANNING, scaled_range, 256)
    #print("shape of range cube is", range_cube.shape)
    doppler_cube = dsp.range_doppler_fft(range_cube[:,:,0:128], utils.Window.HANNING, scaled_doppler, 64)
    doppler_cube = dsp.phase_compensation(doppler_cube, 64, 2)
    doppler_cube = dsp.hadamard2_decode_openradar(doppler_cube)
    doppler_map = np.sum(np.abs(doppler_cube), 1)
    axes[0].imshow(doppler_map, aspect=0.5, interpolation=None)
    det_obj = cfar_test(doppler_map)
    print(det_obj)
    axes[1].imshow(det_obj, aspect=0.5, interpolation=None)
    plt.pause(.5)
